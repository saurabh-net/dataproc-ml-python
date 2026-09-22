# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Retry policy and error reporting for the AI functions.

Transient backend failures are retried with full-jitter exponential backoff.
Errors that survive the retry budget are reported per row in the ``status``
field of the result, so that a single bad row never fails an entire Spark
stage.
"""

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# HTTP statuses that are worth retrying: rate limiting, transient backend
# failures and gateway timeouts.
RETRYABLE_HTTP_CODES = frozenset({408, 429, 500, 503, 504})

# Canonical error codes whose messages BigQuery passes through verbatim; these
# describe a problem with the request itself, so the detail is actionable.
_PASSTHROUGH_CODES = frozenset(
    {
        "FAILED_PRECONDITION",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "INVALID_ARGUMENT",
        "OUT_OF_RANGE",
    }
)

_QUOTA_MESSAGE_PREFIX = "Quota exceeded for metric "

_HTTP_TO_CANONICAL_CODE = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    408: "DEADLINE_EXCEEDED",
    409: "ALREADY_EXISTS",
    413: "OUT_OF_RANGE",
    429: "RESOURCE_EXHAUSTED",
    499: "CANCELLED",
    500: "INTERNAL",
    501: "NOT_IMPLEMENTED",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}

#: Reported when the request was throttled and outlived the retry budget.
RATE_LIMITED_STATUS = "RATE_LIMITED"


def _api_error_types():
    """Returns the google-genai API error types, imported lazily."""
    from google.genai import errors  # pylint: disable=import-outside-toplevel

    return errors.APIError


def _transport_error_types():
    """Returns the transport error types treated as retryable."""
    import httpx  # pylint: disable=import-outside-toplevel

    return (httpx.TransportError, TimeoutError)


def is_reportable(exception: BaseException) -> bool:
    """Reports whether an exception should become a row-level ``status``.

    Only failures that come from the model backend are reported per row.
    Anything else, such as a bug in this library, is left to propagate so that
    it is not silently hidden in a column of an otherwise successful query.

    Args:
        exception: The exception raised while calling the model.

    Returns:
        True if the exception should be converted into a status message.
    """
    return isinstance(
        exception, (_api_error_types(),) + _transport_error_types()
    )


def is_retryable(exception: BaseException) -> bool:
    """Reports whether a failed request is worth retrying.

    Args:
        exception: The exception raised while calling the model.

    Returns:
        True for rate limiting, transient backend errors and transport errors.
    """
    if isinstance(exception, _transport_error_types()):
        return True
    if isinstance(exception, _api_error_types()):
        return _http_code(exception) in RETRYABLE_HTTP_CODES
    return False


def _http_code(exception: BaseException) -> Optional[int]:
    """Extracts the HTTP status code from an API error, if it has one."""
    code = getattr(exception, "code", None)
    return code if isinstance(code, int) else None


def _canonical_code(exception: BaseException) -> str:
    """Determines the canonical error code, e.g. ``RESOURCE_EXHAUSTED``."""
    status = getattr(exception, "status", None)
    if isinstance(status, str) and status:
        return status
    code = _http_code(exception)
    if code is not None and code in _HTTP_TO_CANONICAL_CODE:
        return _HTTP_TO_CANONICAL_CODE[code]
    if isinstance(exception, _transport_error_types()):
        return "UNAVAILABLE"
    return "UNKNOWN"


def to_status(exception: BaseException) -> str:
    """Converts a model error into the row's ``status`` value.

    Rate limiting gets its own name because it is the one failure a caller is
    expected to react to programmatically, by slowing the query down or
    retrying it later. Everything else is described rather than classified,
    since the detail is what makes it actionable.

    Args:
        exception: The exception raised while calling the model.

    Returns:
        ``RATE_LIMITED``, or a human readable description. Never empty.
    """
    if _canonical_code(exception) == "RESOURCE_EXHAUSTED":
        return RATE_LIMITED_STATUS
    return to_status_message(exception)


def to_status_message(exception: BaseException) -> str:
    """Describes a model error in words.

    Request errors are surfaced verbatim, quota messages are passed through,
    and everything else is summarized as a retryable error.

    Args:
        exception: The exception raised while calling the model.

    Returns:
        A human readable status message. Never empty.
    """
    code = _canonical_code(exception)
    message = getattr(exception, "message", None) or str(exception)

    if code in _PASSTHROUGH_CODES:
        return f"{code}: {message}"
    if code == "RESOURCE_EXHAUSTED" and message.startswith(
        _QUOTA_MESSAGE_PREFIX
    ):
        return message
    return (
        f"A retryable error occurred: {code} error from remote "
        "service/endpoint."
    )


def full_jitter_delay(
    attempt: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    rng: Optional[random.Random] = None,
) -> float:
    """Computes a full-jitter exponential backoff delay.

    The delay is drawn uniformly from ``[0, min(cap, base * 2 ** attempt)]``.
    Randomising the whole interval, rather than only a fraction of it, stops
    the tasks of a large cluster from retrying in lockstep after a shared
    throttling event.

    Args:
        attempt: The zero-based index of the attempt that just failed.
        base_delay_seconds: The delay unit that is doubled on each attempt.
        max_delay_seconds: The upper bound on the exponential term.
        rng: Random source, overridable for tests.

    Returns:
        The number of seconds to sleep before the next attempt.
    """
    ceiling = min(max_delay_seconds, base_delay_seconds * (2**attempt))
    return (rng or random).uniform(0.0, max(0.0, ceiling))
