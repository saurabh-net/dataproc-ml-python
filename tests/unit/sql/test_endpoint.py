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

"""Tests for the endpoint check."""

import unittest

from google.genai import errors

from google.cloud.dataproc_ml.sql import _endpoint


def api_error(code, status, message):
    """Builds an API error the way the client library reports one."""
    return errors.APIError(
        code, {"error": {"code": code, "status": status, "message": message}}
    )


class FakeModels:
    """Records the names it was asked about, and optionally fails."""

    def __init__(self, error=None):
        self.error = error
        self.asked = []

    def get(self, *, model):
        self.asked.append(model)
        if self.error is not None:
            raise self.error
        return object()


class FakeClient:

    def __init__(self, error=None):
        self.models = FakeModels(error)


class TestNormalize(unittest.TestCase):
    """What ``models.get`` can actually be asked about."""

    def test_a_bare_name_is_looked_up_as_is(self):
        self.assertEqual(
            _endpoint.normalize("gemini-3.6-flash"), "gemini-3.6-flash"
        )

    def test_a_name_is_trimmed(self):
        self.assertEqual(_endpoint.normalize("  gemini-3.6-flash  "),
                         "gemini-3.6-flash")

    def test_a_publisher_path_is_looked_up_as_is(self):
        path = "publishers/google/models/gemini-3.6-flash"
        self.assertEqual(_endpoint.normalize(path), path)

    def test_a_qualified_publisher_path_is_shortened(self):
        # models.get turns a projects/... path into a URL that does not
        # exist, so the part it understands is extracted.
        self.assertEqual(
            _endpoint.normalize(
                "projects/p/locations/global/publishers/google/models/m"
            ),
            "publishers/google/models/m",
        )

    def test_any_other_qualified_resource_is_left_alone(self):
        # A tuned model or a deployed endpoint is not in the catalogue, and
        # must not be rejected merely because it cannot be checked.
        for endpoint in [
            "projects/p/locations/us-central1/endpoints/123",
            "projects/p/locations/us-central1/models/456",
        ]:
            with self.subTest(endpoint=endpoint):
                self.assertIsNone(_endpoint.normalize(endpoint))


class TestCheck(unittest.TestCase):

    def setUp(self):
        _endpoint.clear_cache()

    def test_a_known_model_passes(self):
        client = FakeClient()
        _endpoint.check(client, "gemini-3.6-flash", "p", None)
        self.assertEqual(client.models.asked, ["gemini-3.6-flash"])

    def test_an_unknown_model_is_rejected(self):
        client = FakeClient(api_error(404, "NOT_FOUND", "no such model"))
        with self.assertRaises(_endpoint.UnknownEndpointError) as caught:
            _endpoint.check(client, "gemini-nope", "p", None)
        self.assertIn("gemini-nope", str(caught.exception))

    def test_the_error_is_a_value_error(self):
        # So that a caller catching ValueError, as the rest of this package
        # does, still sees it.
        client = FakeClient(api_error(404, "NOT_FOUND", "no such model"))
        with self.assertRaises(ValueError):
            _endpoint.check(client, "gemini-nope", "p", None)

    def test_a_catalogue_outage_does_not_fail_the_query(self):
        # Generation reports its own errors; an unavailable catalogue must not
        # break a query that would have worked.
        client = FakeClient(api_error(503, "UNAVAILABLE", "try later"))
        _endpoint.check(client, "gemini-3.6-flash", "p", None)

    def test_a_model_is_checked_once_per_process(self):
        client = FakeClient()
        for _ in range(3):
            _endpoint.check(client, "gemini-3.6-flash", "p", None)
        self.assertEqual(len(client.models.asked), 1)

    def test_each_model_is_checked_separately(self):
        client = FakeClient()
        _endpoint.check(client, "gemini-3.6-flash", "p", None)
        _endpoint.check(client, "gemini-2.5-pro", "p", None)
        self.assertEqual(
            client.models.asked, ["gemini-3.6-flash", "gemini-2.5-pro"]
        )

    def test_the_cache_is_keyed_by_project_and_location(self):
        client = FakeClient()
        _endpoint.check(client, "gemini-3.6-flash", "p", None)
        _endpoint.check(client, "gemini-3.6-flash", "other", None)
        _endpoint.check(client, "gemini-3.6-flash", "p", "us-central1")
        self.assertEqual(len(client.models.asked), 3)

    def test_an_uncheckable_endpoint_is_not_looked_up(self):
        client = FakeClient()
        _endpoint.check(
            client, "projects/p/locations/l/endpoints/1", "p", None
        )
        self.assertEqual(client.models.asked, [])


if __name__ == "__main__":
    unittest.main()
