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

"""Tests for the multimodal prompt parts."""

import unittest

from google.cloud.dataproc_ml.sql import _parts


class TestFileStructRecognition(unittest.TestCase):
    """Telling a lone file reference from a list of parts."""

    def test_the_two_file_shapes(self):
        for names in [["uri"], ["uri", "content_type"],
                      ["content_type", "uri"]]:
            with self.subTest(names=names):
                self.assertTrue(_parts.is_file_struct(names))

    def test_anything_else_is_a_part_list(self):
        for names in [
            [],
            ["col1"],
            ["uri", "col2"],
            ["uri", "content_type", "size"],
            ["url"],
        ]:
            with self.subTest(names=names):
                self.assertFalse(_parts.is_file_struct(names))


class TestMimeTypes(unittest.TestCase):

    def test_common_extensions(self):
        for uri, expected in [
            ("gs://b/invoice.pdf", "application/pdf"),
            ("gs://b/photo.JPG", "image/jpeg"),
            ("gs://b/photo.png", "image/png"),
            ("gs://b/clip.mp4", "video/mp4"),
            ("gs://b/notes.txt", "text/plain"),
        ]:
            with self.subTest(uri=uri):
                self.assertEqual(_parts.guess_mime_type(uri), expected)

    def test_extensions_python_does_not_know(self):
        # Gemini accepts these, so a user must not have to spell them out.
        for uri, expected in [
            ("gs://b/photo.heic", "image/heic"),
            ("gs://b/song.flac", "audio/flac"),
        ]:
            with self.subTest(uri=uri):
                self.assertEqual(_parts.guess_mime_type(uri), expected)

    def test_a_dot_in_the_bucket_is_not_an_extension(self):
        with self.assertRaises(ValueError):
            _parts.guess_mime_type("gs://my.bucket.name/object")

    def test_a_query_string_is_ignored(self):
        self.assertEqual(
            _parts.guess_mime_type("https://h/x.pdf?generation=17"),
            "application/pdf",
        )

    def test_an_unknown_name_asks_for_the_type(self):
        with self.assertRaises(ValueError) as caught:
            _parts.guess_mime_type("gs://b/mystery")
        self.assertIn("content_type", str(caught.exception))


class TestFileParts(unittest.TestCase):

    def test_a_uri_alone(self):
        part = _parts.to_file_part({"uri": "gs://b/x.pdf"})
        self.assertEqual(part.file_data.file_uri, "gs://b/x.pdf")
        self.assertEqual(part.file_data.mime_type, "application/pdf")

    def test_the_content_type_wins_over_the_extension(self):
        part = _parts.to_file_part(
            {"uri": "gs://b/x.pdf", "content_type": "text/plain"}
        )
        self.assertEqual(part.file_data.mime_type, "text/plain")

    def test_a_null_content_type_falls_back_to_the_extension(self):
        part = _parts.to_file_part(
            {"uri": "gs://b/x.pdf", "content_type": None}
        )
        self.assertEqual(part.file_data.mime_type, "application/pdf")

    def test_the_uri_is_trimmed(self):
        part = _parts.to_file_part({"uri": "  gs://b/x.pdf  "})
        self.assertEqual(part.file_data.file_uri, "gs://b/x.pdf")

    def test_a_missing_uri_is_rejected(self):
        for value in [{}, {"uri": None}, {"uri": ""}, {"uri": "   "}]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _parts.to_file_part(value)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError) as caught:
            _parts.to_file_part({"uri": "gs://b/x.pdf", "size": 1})
        self.assertIn("size", str(caught.exception))


class TestContents(unittest.TestCase):
    """Turning one row's fields into the contents of a request."""

    def test_text_and_files_keep_their_order(self):
        contents = list(
            _parts.to_contents(["before", {"uri": "gs://b/x.pdf"}, "after"])
        )
        self.assertEqual(
            [
                part.text if part.text is not None else part.file_data.file_uri
                for part in contents
            ],
            ["before", "gs://b/x.pdf", "after"],
        )

    def test_null_parts_are_skipped(self):
        contents = list(_parts.to_contents([None, "text", float("nan")]))
        self.assertEqual([part.text for part in contents], ["text"])

    def test_an_empty_string_contributes_nothing(self):
        self.assertIsNone(_parts.to_contents(["", ""]))

    def test_a_row_with_nothing_in_it_is_a_null_prompt(self):
        for row in [[], [None], [{"uri": None}], [None, {"uri": None}]]:
            with self.subTest(row=row):
                self.assertIsNone(_parts.to_contents(row))

    def test_a_part_of_the_wrong_type_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            _parts.to_contents(["ok", 42])
        # The position is one-based so that it reads like the query.
        self.assertIn("part 2", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
