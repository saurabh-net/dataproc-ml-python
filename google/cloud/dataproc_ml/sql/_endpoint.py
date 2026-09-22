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

"""Endpoint checking.

A misspelled model name is a mistake in the query, not a property of a row, so
it should stop the query rather than fill a column with failures. The name is
checked once per worker process against the model catalogue, before any
generation is attempted.
"""

import logging
import re
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# projects/<p>/locations/<l>/publishers/<pub>/models/<model>
_QUALIFIED_PUBLISHER_PATH = re.compile(
    r"^projects/[^/]+/locations/[^/]+/(publishers/[^/]+/models/.+)$"
)

_LOCK = threading.Lock()
_CHECKED: Dict[Tuple[Optional[str], Optional[str], str], None] = {}


class UnknownEndpointError(ValueError):
    """Raised when the endpoint does not name a model that can be called."""


def normalize(endpoint: str) -> Optional[str]:
    """Returns the name to look the endpoint up under, or None to skip.

    ``models.get`` resolves a bare name and a ``publishers/.../models/...``
    path, but not a fully qualified ``projects/.../locations/...`` one, which
    it turns into a request to a URL that does not exist. A qualified path that
    names a publisher model is therefore shortened to the part the catalogue
    understands. Anything else qualified, such as a tuned model or an endpoint
    in another project, is left unchecked rather than wrongly rejected.

    Args:
        endpoint: The endpoint as the caller wrote it.

    Returns:
        The name to look up, or None if the endpoint cannot be checked.
    """
    endpoint = endpoint.strip()
    match = _QUALIFIED_PUBLISHER_PATH.match(endpoint)
    if match:
        return match.group(1)
    if endpoint.startswith("projects/"):
        return None
    return endpoint


def check(
    client, endpoint: str, project: Optional[str], location: Optional[str]
) -> None:
    """Verifies once per process that the endpoint names a real model.

    Args:
        client: The ``google.genai.Client`` to ask.
        endpoint: The endpoint as the caller wrote it.
        project: The project the client is configured for, used as a cache key.
        location: The location the client is configured for, likewise.

    Raises:
        UnknownEndpointError: If the model does not exist or is not visible.
    """
    key = (project, location, endpoint)
    with _LOCK:
        if key in _CHECKED:
            return

    name = normalize(endpoint)
    if name is None:
        logger.debug("Not checking the fully qualified endpoint %s.", endpoint)
    else:
        try:
            client.models.get(model=name)
        except Exception as e:  # pylint: disable=broad-except
            if _is_not_found(e):
                raise UnknownEndpointError(
                    f"The endpoint '{endpoint}' does not name a model "
                    "available to this project. Check the spelling, or pass a "
                    "fully qualified "
                    "'projects/.../locations/.../publishers/.../models/...' "
                    f"path. Cause: {e}"
                ) from e
            # A transient failure of the catalogue must not fail a query that
            # would otherwise have worked; generation reports its own errors.
            logger.warning(
                "Could not check the endpoint '%s', continuing: %s", endpoint, e
            )

    with _LOCK:
        _CHECKED[key] = None


def _is_not_found(exception: BaseException) -> bool:
    """Reports whether an error means the model does not exist."""
    if getattr(exception, "code", None) == 404:
        return True
    status = getattr(exception, "status", None)
    return isinstance(status, str) and status.upper() == "NOT_FOUND"


def clear_cache() -> None:
    """Forgets which endpoints have been checked. Intended for tests."""
    with _LOCK:
        _CHECKED.clear()
