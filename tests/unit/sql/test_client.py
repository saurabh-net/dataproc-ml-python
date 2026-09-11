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

import unittest
from unittest import mock

from google.cloud.dataproc_ml.sql import _client


class TestResolveProjectAndLocation(unittest.TestCase):

    def test_explicit_values_win(self):
        env = {
            "GOOGLE_CLOUD_PROJECT": "from-env",
            "GOOGLE_CLOUD_LOCATION": "europe-west4",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            project, location = _client.resolve_project_and_location(
                "explicit-project", "asia-northeast1"
            )

        self.assertEqual(project, "explicit-project")
        self.assertEqual(location, "asia-northeast1")

    def test_project_falls_back_to_the_environment(self):
        with mock.patch.dict(
            "os.environ", {"GOOGLE_CLOUD_PROJECT": "from-env"}, clear=True
        ):
            project, _ = _client.resolve_project_and_location(None, None)

        self.assertEqual(project, "from-env")

    def test_location_defaults_to_global(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            _, location = _client.resolve_project_and_location("p", None)

        self.assertEqual(location, "global")
        self.assertEqual(_client.DEFAULT_LOCATION, "global")

    def test_location_ignores_the_environment(self):
        # A Dataproc cluster sets these to the region it runs in. Honoring
        # them would send the default model to a region that does not serve
        # it, and the failure would be reported per row rather than raised.
        env = {
            "GOOGLE_CLOUD_PROJECT": "p",
            "GOOGLE_CLOUD_LOCATION": "europe-west4",
            "GOOGLE_CLOUD_REGION": "us-central1",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            _, location = _client.resolve_project_and_location(None, None)

        self.assertEqual(location, "global")


if __name__ == "__main__":
    unittest.main()
