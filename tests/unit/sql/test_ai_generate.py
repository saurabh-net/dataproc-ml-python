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

"""Tests for ``ai_generate``."""

import asyncio
import json
import unittest
from unittest import mock

from google.genai import errors, types
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import (
    ArrayType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# pylint: disable=ungrouped-imports
from google.cloud.dataproc_ml.sql import _ai_generate as ai_generate_module
from google.cloud.dataproc_ml.sql import ai_generate, ai_generate_udf

# pylint: enable=ungrouped-imports


def response_with_text(text, thought=None):
    """Builds a model response containing a single answer part."""
    parts = []
    if thought is not None:
        parts.append(types.Part(text=thought, thought=True))
    parts.append(types.Part(text=text))
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=parts))
        ]
    )


def blocked_response(reason=types.FinishReason.SAFETY):
    """Builds a response whose content was withheld rather than generated."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[]),
                finish_reason=reason,
            )
        ]
    )


def api_error(code, status, message):
    """Builds an API error the way the client library reports one."""
    return errors.APIError(
        code, {"error": {"code": code, "status": status, "message": message}}
    )


def replies_with(*outcomes):
    """Returns a fake ``generate_content`` that replays the given outcomes.

    Each outcome is either a response to return or an exception to raise. The
    returned coroutine function is defined here rather than at module level so
    that it is serialized by value and can run inside a Spark worker.

    Args:
        *outcomes: Responses or exceptions, consumed in order. The last one is
            repeated once exhausted.

    Returns:
        An awaitable callable with the same shape as the client method.
    """
    remaining = list(outcomes)

    async def _generate_content(*, model, contents, config):
        del model, contents, config
        outcome = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return _generate_content


def answers_the_prompt():
    """Returns a fake that echoes each prompt, so batching can be checked."""

    async def _generate_content(*, model, contents, config):
        del model, config
        return response_with_text(f"answer to {contents}")

    return _generate_content


def echoes_the_request():
    """Returns a fake that reports the arguments it was called with."""

    async def _generate_content(*, model, contents, config):
        return response_with_text(
            json.dumps(
                {
                    "model": model,
                    "contents": contents,
                    "temperature": getattr(config, "temperature", None),
                    "response_schema": getattr(
                        config, "response_schema", None
                    ),
                },
                default=str,
            )
        )

    return _generate_content


def run(coroutine):
    """Runs one coroutine to completion."""
    return asyncio.run(coroutine)


def generate_one(**kwargs):
    """Calls the per-row entry point with the arguments a batch would pass."""
    kwargs.setdefault("endpoint", None)
    kwargs.setdefault("model_params", None)
    kwargs.setdefault("output_schema", None)
    return run(ai_generate_module._generate_one(**kwargs))


class SparkTestCase(unittest.TestCase):
    """Base class holding a Spark session shared by every test module."""

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("dataproc-ml-sql-tests")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    def setUp(self):
        # The configuration cache is process wide, so it is cleared between
        # tests to keep them independent.
        ai_generate_module._resolve_config.cache_clear()


class TestPromptColumn(SparkTestCase):

    def test_a_string_names_a_column(self):
        df = self.spark.createDataFrame([("hello",)], ["text"])
        column = ai_generate_module._to_prompt_column("text")
        self.assertEqual(df.select(column).first()[0], "hello")

    def test_parts_are_concatenated_in_order(self):
        df = self.spark.createDataFrame([("world",)], ["text"])
        column = ai_generate_module._to_prompt_column(
            [sf.lit("hello "), sf.col("text"), sf.lit("!")]
        )
        self.assertEqual(df.select(column).first()[0], "hello world!")

    def test_a_single_part_is_used_as_is(self):
        df = self.spark.createDataFrame([("hello",)], ["text"])
        column = ai_generate_module._to_prompt_column([sf.col("text")])
        self.assertEqual(df.select(column).first()[0], "hello")

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            ai_generate_module._to_prompt_column([])

    def test_unsupported_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            ai_generate_module._to_prompt_column(42)

    def test_a_non_string_column_is_cast_to_string(self):
        # Without the cast the worker would see an int, treat it as a null
        # prompt and return nulls for the whole column without any error.
        df = self.spark.createDataFrame([(7,)], ["n"])

        for prompt in ("n", sf.col("n"), [sf.col("n")]):
            with self.subTest(prompt=prompt):
                column = ai_generate_module._to_prompt_column(prompt)
                value = df.select(column).first()[0]
                self.assertEqual(value, "7")
                self.assertIsInstance(value, str)


class TestReturnType(SparkTestCase):
    """The return type is fixed, which is what makes the arguments dynamic."""

    def test_fields(self):
        self.assertEqual(
            [field.name for field in ai_generate_module.RETURN_TYPE.fields],
            ["result", "full_response", "status"],
        )
        for field in ai_generate_module.RETURN_TYPE.fields:
            with self.subTest(field=field.name):
                self.assertIsInstance(field.dataType, StringType)

    def test_output_schema_does_not_change_the_return_type(self):
        # The whole point of the fixed shape: a query can be planned without
        # knowing what output_schema will be.
        df = self.spark.createDataFrame([("hi",)], ["text"])
        plain = df.withColumn(
            "g", ai_generate("text", _generate_content=answers_the_prompt())
        )
        structured = df.withColumn(
            "g",
            ai_generate(
                "text",
                output_schema="sentiment STRING, score INT",
                _generate_content=answers_the_prompt(),
            ),
        )
        self.assertEqual(
            plain.schema["g"].dataType, structured.schema["g"].dataType
        )
        self.assertEqual(
            plain.schema["g"].dataType.simpleString(),
            "struct<result:string,full_response:string,status:string>",
        )


class TestArgumentRendering(unittest.TestCase):
    """Arguments are rendered to strings so they can travel as columns."""

    def test_model_params_accepts_a_json_string(self):
        self.assertEqual(
            ai_generate_module._to_model_params_json('{"a": 1}'), '{"a": 1}'
        )

    def test_model_params_accepts_a_dictionary(self):
        rendered = ai_generate_module._to_model_params_json({"a": 1})
        self.assertEqual(json.loads(rendered), {"a": 1})

    def test_model_params_rejects_other_types(self):
        for value in [42, ["a"], object()]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ai_generate_module._to_model_params_json(value)

    def test_model_params_rejects_values_that_are_not_json(self):
        with self.assertRaises(ValueError):
            ai_generate_module._to_model_params_json({"a": {1, 2}})

    def test_output_schema_accepts_a_ddl_string(self):
        self.assertEqual(
            ai_generate_module._to_output_schema_ddl("a STRING"), "a STRING"
        )

    def test_output_schema_accepts_a_struct_type(self):
        rendered = ai_generate_module._to_output_schema_ddl(
            StructType(
                [
                    StructField("a", StringType()),
                    StructField("b", ArrayType(LongType())),
                ]
            )
        )
        # Whatever Spark renders must be readable by our own parser.
        self.assertEqual(
            ai_generate_module._ddl.parse_response_schema(rendered),
            {
                "type": "OBJECT",
                "properties": {
                    "a": {"type": "STRING"},
                    "b": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                },
            },
        )

    def test_output_schema_rejects_other_types(self):
        for value in [42, ["a STRING"]]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ai_generate_module._to_output_schema_ddl(value)

    def test_argument_text_treats_missing_and_blank_alike(self):
        for value in [None, "", "   "]:
            with self.subTest(value=value):
                self.assertIsNone(ai_generate_module._argument_text(value))

    def test_argument_text_trims(self):
        self.assertEqual(
            ai_generate_module._argument_text("  gemini-3.6-flash  "),
            "gemini-3.6-flash",
        )


class TestConfigResolution(unittest.TestCase):
    """Turning the string arguments into a generation config."""

    def setUp(self):
        ai_generate_module._resolve_config.cache_clear()

    def test_output_schema_becomes_a_constrained_response_schema(self):
        config = ai_generate_module._resolve_config(
            None, "sentiment STRING, score INT"
        )
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(
            config.response_schema,
            {
                "type": "OBJECT",
                "properties": {
                    "sentiment": {"type": "STRING"},
                    "score": {"type": "INTEGER"},
                },
            },
        )

    def test_model_params_are_unwrapped_into_the_config(self):
        config = ai_generate_module._resolve_config(
            '{"generation_config": {"temperature": 0.25,'
            ' "max_output_tokens": 64}}',
            None,
        )
        self.assertEqual(config.temperature, 0.25)
        self.assertEqual(config.max_output_tokens, 64)

    def test_json_numbers_stay_floats(self):
        # A JSON string decodes to plain floats. The same value arriving as a
        # Spark named_struct would be a Decimal, which is not serializable.
        config = ai_generate_module._resolve_config(
            '{"generation_config": {"temperature": 0.7}}', None
        )
        self.assertIsInstance(config.temperature, float)

    def test_no_arguments_gives_an_empty_config(self):
        config = ai_generate_module._resolve_config(None, None)
        self.assertIsNone(config.response_schema)
        self.assertIsNone(config.temperature)

    def test_the_same_arguments_are_resolved_once(self):
        first = ai_generate_module._resolve_config(None, "a STRING")
        second = ai_generate_module._resolve_config(None, "a STRING")
        self.assertIs(first, second)

    def test_different_arguments_resolve_separately(self):
        first = ai_generate_module._resolve_config(None, "a STRING")
        second = ai_generate_module._resolve_config(None, "b INT")
        self.assertIsNot(first, second)

    def test_invalid_arguments_are_rejected(self):
        for model_params, output_schema in [
            ("{not json", None),
            ('{"contents": []}', None),
            (None, "a NOTATYPE"),
            ('{"generation_config": {"temperature": "hot"}}', None),
        ]:
            with self.subTest(
                model_params=model_params, output_schema=output_schema
            ):
                with self.assertRaises(ValueError):
                    ai_generate_module._resolve_config(
                        model_params, output_schema
                    )


class TestGenerateOneRow(unittest.TestCase):
    """The per-row path, including every value ``status`` can take."""

    def setUp(self):
        ai_generate_module._resolve_config.cache_clear()

    def test_a_successful_row(self):
        row = generate_one(
            generate_content=replies_with(response_with_text("hello")),
            prompt="hi",
        )
        self.assertEqual(row["result"], "hello")
        self.assertEqual(row["status"], "SUCCESS")
        self.assertIn("candidates", row["full_response"])

    def test_full_response_is_a_json_string(self):
        row = generate_one(
            generate_content=replies_with(response_with_text("hello")),
            prompt="hi",
        )
        self.assertIsInstance(row["full_response"], str)
        self.assertIsInstance(json.loads(row["full_response"]), dict)

    def test_a_null_prompt_is_not_sent_to_the_model(self):
        def _fail(**_):
            raise AssertionError("the model must not be called")

        for prompt in [None, float("nan")]:
            with self.subTest(prompt=prompt):
                row = generate_one(generate_content=_fail, prompt=prompt)
                self.assertIsNone(row["result"])
                self.assertIsNone(row["full_response"])
                self.assertEqual(row["status"], "SUCCESS")

    def test_reasoning_parts_are_skipped(self):
        row = generate_one(
            generate_content=replies_with(
                response_with_text("the answer", thought="thinking...")
            ),
            prompt="hi",
        )
        self.assertEqual(row["result"], "the answer")

    def test_blocked_content_is_reported(self):
        for reason in [
            types.FinishReason.SAFETY,
            types.FinishReason.PROHIBITED_CONTENT,
        ]:
            with self.subTest(reason=reason):
                row = generate_one(
                    generate_content=replies_with(blocked_response(reason)),
                    prompt="hi",
                )
                self.assertIsNone(row["result"])
                self.assertEqual(row["status"], "SAFETY_BLOCKED")
                # The reason is still recoverable from the full response.
                self.assertIsNotNone(row["full_response"])

    def test_an_ordinary_finish_reason_is_not_a_block(self):
        row = generate_one(
            generate_content=replies_with(
                types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            content=types.Content(
                                role="model", parts=[types.Part(text="hi")]
                            ),
                            finish_reason=types.FinishReason.MAX_TOKENS,
                        )
                    ]
                )
            ),
            prompt="hi",
        )
        self.assertEqual(row["status"], "SUCCESS")
        self.assertEqual(row["result"], "hi")

    def test_rate_limiting_is_reported(self):
        with mock.patch.object(ai_generate_module, "_MAX_ATTEMPTS", 1):
            row = generate_one(
                generate_content=replies_with(
                    api_error(429, "RESOURCE_EXHAUSTED", "slow down")
                ),
                prompt="hi",
            )
        self.assertIsNone(row["result"])
        self.assertEqual(row["status"], "RATE_LIMITED")

    def test_a_request_error_is_described(self):
        with mock.patch.object(ai_generate_module, "_MAX_ATTEMPTS", 1):
            row = generate_one(
                generate_content=replies_with(
                    api_error(404, "NOT_FOUND", "no such model")
                ),
                prompt="hi",
            )
        self.assertIsNone(row["result"])
        self.assertIn("NOT_FOUND", row["status"])

    def test_a_transient_failure_is_retried(self):
        with mock.patch.object(
            ai_generate_module, "_BASE_RETRY_DELAY_SECONDS", 0.0
        ):
            row = generate_one(
                generate_content=replies_with(
                    api_error(503, "UNAVAILABLE", "try again"),
                    response_with_text("recovered"),
                ),
                prompt="hi",
            )
        self.assertEqual(row["result"], "recovered")
        self.assertEqual(row["status"], "SUCCESS")

    def test_a_bug_in_this_library_is_not_hidden_in_status(self):
        async def _raise(**_):
            raise TypeError("this is a bug, not a backend failure")

        with self.assertRaises(TypeError):
            generate_one(generate_content=_raise, prompt="hi")

    def test_an_invalid_argument_is_reported_per_row(self):
        # Called from SQL the arguments are columns, so they cannot be
        # rejected on the driver.
        row = generate_one(
            generate_content=replies_with(response_with_text("unused")),
            prompt="hi",
            output_schema="a NOTATYPE",
        )
        self.assertIsNone(row["result"])
        self.assertIn("output_schema", row["status"])

    def test_the_endpoint_argument_selects_the_model(self):
        row = generate_one(
            generate_content=echoes_the_request(),
            prompt="hi",
            endpoint="gemini-2.5-pro",
        )
        self.assertEqual(json.loads(row["result"])["model"], "gemini-2.5-pro")

    def test_a_missing_endpoint_falls_back_to_the_default(self):
        for endpoint in [None, "", "   "]:
            with self.subTest(endpoint=endpoint):
                row = generate_one(
                    generate_content=echoes_the_request(),
                    prompt="hi",
                    endpoint=endpoint,
                )
                self.assertEqual(
                    json.loads(row["result"])["model"],
                    ai_generate_module.DEFAULT_ENDPOINT,
                )

    def test_a_fully_qualified_endpoint_is_passed_through(self):
        path = "projects/p/locations/l/publishers/google/models/m"
        row = generate_one(
            generate_content=echoes_the_request(), prompt="hi", endpoint=path
        )
        self.assertEqual(json.loads(row["result"])["model"], path)

    def test_a_non_string_prompt_is_rendered(self):
        # Arrow delivers a nullable BIGINT column as floats, so a whole number
        # must not pick up a fraction just because another row was null.
        row = generate_one(
            generate_content=answers_the_prompt(), prompt=2.0
        )
        self.assertEqual(row["result"], "answer to 2")


class TestDriverValidation(SparkTestCase):
    """The DataFrame entry point rejects bad arguments before the query runs."""

    def test_rejects_an_empty_endpoint(self):
        for endpoint in ["", "   ", None, 42]:
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(ValueError):
                    ai_generate("text", endpoint=endpoint)

    def test_rejects_an_invalid_output_schema(self):
        with self.assertRaises(ValueError):
            ai_generate("text", output_schema="a NOTATYPE")

    def test_rejects_invalid_model_params(self):
        with self.assertRaises(ValueError):
            ai_generate("text", model_params="{not json")

    def test_rejects_model_params_carrying_contents(self):
        with self.assertRaises(ValueError):
            ai_generate("text", model_params='{"contents": []}')

    def test_rejects_two_response_schemas(self):
        with self.assertRaises(ValueError):
            ai_generate(
                "text",
                model_params='{"generation_config":'
                ' {"response_schema": {"type": "OBJECT"}}}',
                output_schema="a STRING",
            )


class TestSparkExecution(SparkTestCase):
    """End to end through a real Spark plan, with the backend faked out."""

    def _documents(self):
        return self.spark.createDataFrame(
            [(1, "alpha"), (2, "beta"), (3, "gamma"), (4, None)],
            "id INT, body STRING",
        )

    def test_every_row_gets_its_own_answer(self):
        rows = (
            self._documents()
            .withColumn(
                "g", ai_generate("body", _generate_content=answers_the_prompt())
            )
            .select("id", "g.result", "g.status")
            .orderBy("id")
            .collect()
        )

        self.assertEqual(
            [(r["id"], r["result"]) for r in rows],
            [
                (1, "answer to alpha"),
                (2, "answer to beta"),
                (3, "answer to gamma"),
                (4, None),
            ],
        )
        for row in rows:
            self.assertEqual(row["status"], "SUCCESS")

    def test_a_failing_row_does_not_fail_the_query(self):
        with mock.patch.object(ai_generate_module, "_MAX_ATTEMPTS", 1):
            rows = (
                self._documents()
                .withColumn(
                    "g",
                    ai_generate(
                        "body",
                        _generate_content=replies_with(
                            api_error(404, "NOT_FOUND", "no such model")
                        ),
                    ),
                )
                .select("g.result", "g.status")
                .collect()
            )

        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertIsNone(row["result"])

    def test_full_response_can_be_parsed_back_into_json(self):
        # The spec fixes the field as a string; parse_json recovers a VARIANT
        # for anyone who wants to query inside it.
        result = (
            self._documents()
            .limit(1)
            .withColumn(
                "g", ai_generate("body", _generate_content=answers_the_prompt())
            )
            .selectExpr("variant_get(parse_json(g.full_response),"
                        " '$.candidates[0].content.role', 'string') AS role")
        )
        self.assertEqual(result.first()["role"], "model")

    def test_registered_for_spark_sql_with_named_arguments(self):
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=echoes_the_request()),
        )
        self._documents().createOrReplaceTempView("documents")

        row = self.spark.sql(
            "SELECT ai_generate("
            "  prompt => body,"
            "  endpoint => 'gemini-2.5-pro',"
            "  output_schema => 'sentiment STRING'"
            ").result AS r FROM documents WHERE id = 1"
        ).first()

        request = json.loads(row["r"])
        self.assertEqual(request["model"], "gemini-2.5-pro")
        self.assertEqual(request["contents"], "alpha")
        self.assertEqual(
            request["response_schema"],
            {
                "type": "OBJECT",
                "properties": {"sentiment": {"type": "STRING"}},
            },
        )

    def test_named_arguments_may_be_reordered_and_omitted(self):
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=echoes_the_request()),
        )
        self._documents().createOrReplaceTempView("documents")

        row = self.spark.sql(
            "SELECT ai_generate(output_schema => 'a STRING', prompt => body)"
            ".result AS r FROM documents WHERE id = 1"
        ).first()

        request = json.loads(row["r"])
        self.assertEqual(
            request["model"], ai_generate_module.DEFAULT_ENDPOINT
        )
        self.assertIsNotNone(request["response_schema"])

    def test_a_bad_argument_from_sql_is_reported_per_row(self):
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=answers_the_prompt()),
        )
        self._documents().createOrReplaceTempView("documents")

        rows = self.spark.sql(
            "SELECT ai_generate(prompt => body,"
            " output_schema => 'a NOTATYPE').status AS s FROM documents"
        ).collect()

        self.assertEqual(len(rows), 4)
        # The null-prompt row never reaches the configuration.
        for row in rows[:3]:
            self.assertIn("output_schema", row["s"])

    def test_the_argument_may_vary_per_row(self):
        # A column, not a literal: each row must get its own configuration.
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=echoes_the_request()),
        )
        self.spark.createDataFrame(
            [("a", "gemini-2.5-pro"), ("b", "gemini-3.6-flash")],
            "body STRING, model STRING",
        ).createOrReplaceTempView("varying")

        rows = self.spark.sql(
            "SELECT body, ai_generate(prompt => body, endpoint => model).result"
            " AS r FROM varying ORDER BY body"
        ).collect()

        self.assertEqual(
            [json.loads(r["r"])["model"] for r in rows],
            ["gemini-2.5-pro", "gemini-3.6-flash"],
        )

    def test_a_non_string_column_works_in_spark_sql(self):
        # Spark SQL inserts no cast, so the worker has to render the value.
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=answers_the_prompt()),
        )
        self._documents().createOrReplaceTempView("documents")

        rows = self.spark.sql(
            "SELECT ai_generate(id).result AS r FROM documents ORDER BY id"
        ).collect()

        self.assertEqual(
            [r["r"] for r in rows],
            ["answer to 1", "answer to 2", "answer to 3", "answer to 4"],
        )


if __name__ == "__main__":
    unittest.main()
