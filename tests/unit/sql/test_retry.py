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
    """``status`` names rate limiting and describes everything else."""

    def test_rate_limiting_gets_its_own_name(self):
        for error in [
            _api_error(429, "RESOURCE_EXHAUSTED", "too many requests"),
            _api_error(429, "", "too many requests"),
        ]:
            with self.subTest(error=error):
                self.assertEqual(_retry.to_status(error), "RATE_LIMITED")

    def test_quota_exhaustion_is_rate_limiting(self):
        message = (
            "Quota exceeded for metric aiplatform.googleapis.com/"
            "generate_content_requests_per_minute_per_project"
        )
        error = _api_error(429, "RESOURCE_EXHAUSTED", message)
        self.assertEqual(_retry.to_status(error), "RATE_LIMITED")

    def test_other_failures_are_described_rather_than_classified(self):
        error = _api_error(404, "NOT_FOUND", "Publisher Model was not found")
        self.assertEqual(
            _retry.to_status(error),
            "NOT_FOUND: Publisher Model was not found",
        )

    def test_transport_failures_are_described(self):
        status = _retry.to_status(httpx.ConnectError("no route"))
        self.assertNotEqual(status, "RATE_LIMITED")
        self.assertIn("UNAVAILABLE", status)

    def test_status_is_never_empty(self):
        self.assertTrue(_retry.to_status(_api_error(418, "", "")))


class TestStatusMessages(unittest.TestCase):
    """Failures that are not rate limiting are described in words."""

    def test_request_errors_are_surfaced_verbatim(self):
        error = _api_error(
            400,
            "INVALID_ARGUMENT",
            "HTTP links are not supported for requests restricted by VPCSC.",
        )
        self.assertEqual(
            _retry.to_status_message(error),
            "INVALID_ARGUMENT: HTTP links are not supported for requests "
            "restricted by VPCSC.",
        )

    def test_not_found_is_surfaced_verbatim(self):
        error = _api_error(404, "NOT_FOUND", "Publisher Model was not found")
        self.assertEqual(
            _retry.to_status_message(error),
            "NOT_FOUND: Publisher Model was not found",
        )

    def test_quota_messages_are_passed_through(self):
        message = (
            "Quota exceeded for metric aiplatform.googleapis.com/"
            "generate_content_requests_per_minute_per_project"
        )
        error = _api_error(429, "RESOURCE_EXHAUSTED", message)
        self.assertEqual(_retry.to_status_message(error), message)

    def test_other_errors_are_summarised_as_retryable(self):
        error = _api_error(429, "RESOURCE_EXHAUSTED", "too many requests")
        self.assertEqual(
            _retry.to_status_message(error),
            "A retryable error occurred: RESOURCE_EXHAUSTED error from remote "
            "service/endpoint.",
        )

    def test_server_errors_are_summarised_as_retryable(self):
        error = _api_error(503, "UNAVAILABLE", "backend down")
        self.assertEqual(
            _retry.to_status_message(error),
            "A retryable error occurred: UNAVAILABLE error from remote "
            "service/endpoint.",
        )

    def test_transport_errors_report_unavailable(self):
        self.assertEqual(
            _retry.to_status_message(httpx.ConnectError("no route")),
            "A retryable error occurred: UNAVAILABLE error from remote "
            "service/endpoint.",
        )

    def test_status_is_never_empty(self):
        error = _api_error(418, "", "")
        self.assertTrue(_retry.to_status_message(error))


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
