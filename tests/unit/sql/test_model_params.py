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

from google.cloud.dataproc_ml.sql import _model_params


class TestNormalizeModelParams(unittest.TestCase):

    def test_no_params_produces_empty_config(self):
        self.assertEqual(_model_params.normalize_model_params(None), {})

    def test_generation_config_is_flattened(self):
        flat = _model_params.normalize_model_params(
            {"generation_config": {"temperature": 0.2, "top_p": 0.9}}
        )
        self.assertEqual(flat, {"temperature": 0.2, "top_p": 0.9})

    def test_camel_case_keys_are_accepted(self):
        flat = _model_params.normalize_model_params(
            {"generationConfig": {"maxOutputTokens": 128, "topK": 3}}
        )
        self.assertEqual(flat, {"max_output_tokens": 128, "top_k": 3})

    def test_nested_thinking_config_is_preserved(self):
        flat = _model_params.normalize_model_params(
            {"generation_config": {"thinking_config": {"thinking_budget": 0}}}
        )
        self.assertEqual(flat, {"thinking_config": {"thinking_budget": 0}})

    def test_top_level_request_fields_are_kept(self):
        flat = _model_params.normalize_model_params(
            {
                "system_instruction": "Answer in one word.",
                "generation_config": {"temperature": 0.0},
            }
        )
        self.assertEqual(
            flat,
            {
                "system_instruction": "Answer in one word.",
                "temperature": 0.0,
            },
        )

    def test_contents_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _model_params.normalize_model_params({"contents": ["hello"]})
        self.assertIn("should not contain contents", str(ctx.exception))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _model_params.normalize_model_params(
                {"generation_config": {"temperature": "warm"}}
            )
        self.assertIn("model_params is invalid", str(ctx.exception))

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaises(ValueError):
            _model_params.normalize_model_params({"not_a_real_field": 1})

    def test_non_dictionary_is_rejected(self):
        with self.assertRaises(ValueError):
            _model_params.normalize_model_params("temperature=0")

    def test_non_dictionary_generation_config_is_rejected(self):
        with self.assertRaises(ValueError):
            _model_params.normalize_model_params({"generation_config": 1})


class TestResponseSchemaHandling(unittest.TestCase):

    def test_output_schema_sets_json_mime_type(self):
        schema = {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}
        flat = _model_params.normalize_model_params(None, schema)
        self.assertEqual(flat["response_mime_type"], "application/json")
        self.assertEqual(flat["response_schema"], schema)

    def test_declaring_a_schema_twice_is_rejected(self):
        schema = {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}
        with self.assertRaises(ValueError) as ctx:
            _model_params.normalize_model_params(
                {"generation_config": {"response_schema": schema}}, schema
            )
        self.assertIn(
            "can only be set either in the model_params or the output_schema",
            str(ctx.exception),
        )

    def test_response_schema_alone_is_allowed(self):
        schema = {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}
        flat = _model_params.normalize_model_params(
            {"generation_config": {"response_schema": schema}}
        )
        self.assertEqual(flat["response_schema"], schema)


class TestBuildGenerationConfig(unittest.TestCase):

    def test_values_reach_the_generation_config(self):
        config = _model_params.build_generation_config(
            {"temperature": 0.25, "max_output_tokens": 64}
        )
        self.assertEqual(config.temperature, 0.25)
        self.assertEqual(config.max_output_tokens, 64)

    def test_thinking_config_is_understood(self):
        config = _model_params.build_generation_config(
            {"thinking_config": {"thinking_budget": 0}}
        )
        self.assertEqual(config.thinking_config.thinking_budget, 0)


if __name__ == "__main__":
    unittest.main()
