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

import random
import unittest

import httpx
from google.genai import errors

from google.cloud.dataproc_ml.sql import _retry


def _api_error(code, status, message):
    """Builds an API error the way the client library reports one."""
    return errors.APIError(
        code, {"error": {"code": code, "status": status, "message": message}}
    )


class TestErrorClassification(unittest.TestCase):

    def test_rate_limiting_and_transient_errors_are_retryable(self):
        for code in (408, 429, 500, 503, 504):
            with self.subTest(code=code):
                error = _api_error(code, "UNAVAILABLE", "try again")
                self.assertTrue(_retry.is_retryable(error))

    def test_request_errors_are_not_retryable(self):
        for code in (400, 401, 403, 404):
            with self.subTest(code=code):
                error = _api_error(code, "INVALID_ARGUMENT", "bad request")
                self.assertFalse(_retry.is_retryable(error))

    def test_transport_errors_are_retryable(self):
        self.assertTrue(_retry.is_retryable(httpx.ConnectError("no route")))
        self.assertTrue(_retry.is_retryable(httpx.ReadTimeout("slow")))

    def test_unrelated_exceptions_are_not_reportable(self):
        # A bug in the library must fail the task rather than be hidden in the
        # status column of an otherwise successful query.
        self.assertFalse(_retry.is_reportable(KeyError("bug")))
        self.assertFalse(_retry.is_retryable(KeyError("bug")))

    def test_backend_errors_are_reportable(self):
        self.assertTrue(
            _retry.is_reportable(_api_error(404, "NOT_FOUND", "no model"))
        )
        self.assertTrue(_retry.is_reportable(httpx.ConnectError("no route")))


class TestStatus(unittest.TestCase):
    """``status`` leads with the canonical code and describes the failure.

    Success is the empty string, so any non-empty status is a failure and the
    dead-letter query never has to know a vocabulary. The code is a prefix
    rather than the whole value so that a caller can still classify a failure
    without this library maintaining a closed taxonomy of its own.
    """

    def test_the_canonical_code_leads(self):
        for code, status in [
            (400, "INVALID_ARGUMENT"),
            (403, "PERMISSION_DENIED"),
            (404, "NOT_FOUND"),
            (429, "RESOURCE_EXHAUSTED"),
            (503, "UNAVAILABLE"),
        ]:
            with self.subTest(status=status):
                error = _api_error(code, status, "the detail")
                self.assertTrue(
                    _retry.to_status(error).startswith(f"{status}: ")
                )

    def test_the_code_is_derived_from_the_http_status_when_absent(self):
        # The backend does not always fill in the canonical status.
        self.assertTrue(
            _retry.to_status(_api_error(429, "", "too many")).startswith(
                "RESOURCE_EXHAUSTED: "
            )
        )

    def test_actionable_failures_are_surfaced_verbatim(self):
        for code, status, message in [
            (
                400,
                "INVALID_ARGUMENT",
                "HTTP links are not supported for requests restricted by "
                "VPCSC.",
            ),
            (404, "NOT_FOUND", "Publisher Model was not found"),
            (
                429,
                "RESOURCE_EXHAUSTED",
                "Quota exceeded for metric aiplatform.googleapis.com/"
                "generate_content_requests_per_minute_per_project",
            ),
        ]:
            with self.subTest(status=status):
                error = _api_error(code, status, message)
                self.assertEqual(
                    _retry.to_status(error), f"{status}: {message}"
                )

    def test_opaque_failures_are_summarised(self):
        # The remote message for these is usually an internal identifier.
        error = _api_error(503, "UNAVAILABLE", "0x8badf00d")
        self.assertEqual(
            _retry.to_status(error),
            "UNAVAILABLE: a retryable error occurred calling the remote "
            "service.",
        )
        self.assertNotIn("0x8badf00d", _retry.to_status(error))

    def test_transport_failures_report_unavailable(self):
        self.assertTrue(
            _retry.to_status(httpx.ConnectError("no route")).startswith(
                "UNAVAILABLE: "
            )
        )

    def test_an_unmapped_failure_is_still_described(self):
        status = _retry.to_status(_api_error(418, "", ""))
        self.assertTrue(status)
        self.assertTrue(status.startswith("UNKNOWN: "))


class TestFullJitterBackoff(unittest.TestCase):

    def test_delay_grows_exponentially_and_is_fully_randomized(self):
        rng = random.Random(7)
        for attempt in range(5):
            with self.subTest(attempt=attempt):
                ceiling = min(60.0, 1.0 * (2**attempt))
                samples = [
                    _retry.full_jitter_delay(attempt, 1.0, 60.0, rng)
                    for _ in range(200)
                ]
                self.assertTrue(all(0.0 <= s <= ceiling for s in samples))
                # Full jitter draws from the whole interval, so samples should
                # reach both ends rather than cluster near the ceiling.
                self.assertLess(min(samples), ceiling * 0.2)
                self.assertGreater(max(samples), ceiling * 0.8)

    def test_delay_is_capped(self):
        rng = random.Random(1)
        samples = [
            _retry.full_jitter_delay(20, 1.0, 30.0, rng) for _ in range(100)
        ]
        self.assertTrue(all(0.0 <= s <= 30.0 for s in samples))

    def test_zero_base_delay_produces_no_wait(self):
        self.assertEqual(_retry.full_jitter_delay(3, 0.0, 60.0), 0.0)


if __name__ == "__main__":
    unittest.main()
