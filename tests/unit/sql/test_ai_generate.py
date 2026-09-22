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
from google.cloud.dataproc_ml.sql import ai_generate, ai_generate_udf, file

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


def echoes_the_parts():
    """Returns a fake that reports the parts a multimodal prompt became."""

    async def _generate_content(*, model, contents, config):
        del model, config
        if isinstance(contents, str):
            return response_with_text(json.dumps([{"text": contents}]))
        described = []
        for part in contents:
            if part.text is not None:
                described.append({"text": part.text})
            else:
                described.append(
                    {
                        "uri": part.file_data.file_uri,
                        "mime_type": part.file_data.mime_type,
                    }
                )
        return response_with_text(json.dumps(described))

    return _generate_content


def run(coroutine):
    """Runs one coroutine to completion."""
    return asyncio.run(coroutine)


def generate_one(**kwargs):
    """Calls the per-row entry point with the arguments a batch would pass."""
    kwargs.setdefault("model", ai_generate_module.DEFAULT_ENDPOINT)
    kwargs.setdefault("config", ai_generate_module._resolve_config(None, None))
    # pylint: disable-next=missing-kwoa
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

    def test_text_parts_are_concatenated_in_order(self):
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

    def test_a_lone_file_becomes_a_uri_struct(self):
        # The field names matter here: they are what tells the worker that the
        # struct is one file rather than a list of parts.
        df = self.spark.createDataFrame([("gs://b/x.pdf",)], ["uri"])
        column = ai_generate_module._to_prompt_column(file(sf.col("uri")))

        field = df.select(column.alias("p")).schema["p"]
        self.assertEqual(
            [f.name for f in field.dataType.fields], ["uri"]
        )
        self.assertEqual(df.select(column).first()[0]["uri"], "gs://b/x.pdf")

    def test_a_file_carries_its_content_type(self):
        df = self.spark.createDataFrame([("gs://b/x",)], ["uri"])
        column = ai_generate_module._to_prompt_column(
            file(sf.col("uri"), "application/pdf")
        )

        row = df.select(column).first()[0]
        self.assertEqual(row["uri"], "gs://b/x")
        self.assertEqual(row["content_type"], "application/pdf")

    def test_a_mixed_prompt_becomes_a_struct_of_parts(self):
        df = self.spark.createDataFrame([("gs://b/x.pdf",)], ["uri"])
        column = ai_generate_module._to_prompt_column(
            [sf.lit("Read this: "), file(sf.col("uri"))]
        )

        field = df.select(column.alias("p")).schema["p"]
        # Positional names, so that a part can never be mistaken for a file
        # and so that the worker is never tempted to read them.
        self.assertEqual([f.name for f in field.dataType.fields], ["_1", "_2"])
        row = df.select(column).first()[0]
        self.assertEqual(row["_1"], "Read this: ")
        self.assertEqual(row["_2"]["uri"], "gs://b/x.pdf")


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


class TestConstantArguments(unittest.TestCase):
    """The settings configure the call, so they may not vary by row."""

    def _series(self, *values):
        import pandas as pd  # pylint: disable=import-outside-toplevel

        return pd.Series(list(values), dtype="object")

    def test_an_absent_argument_has_no_value(self):
        self.assertIsNone(ai_generate_module._constant(None, "endpoint"))

    def test_an_empty_batch_has_no_value(self):
        self.assertIsNone(
            ai_generate_module._constant(self._series(), "endpoint")
        )

    def test_a_repeated_value_is_the_value(self):
        self.assertEqual(
            ai_generate_module._constant(
                self._series("gemini-3.6-flash", "gemini-3.6-flash"),
                "endpoint",
            ),
            "gemini-3.6-flash",
        )

    def test_nulls_and_blanks_mean_the_default(self):
        for values in [(None, None), ("", ""), ("  ", "  ")]:
            with self.subTest(values=values):
                self.assertIsNone(
                    ai_generate_module._constant(
                        self._series(*values), "endpoint"
                    )
                )

    def test_a_varying_value_is_rejected(self):
        for name in ai_generate_module._CONSTANT_ARGUMENTS:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    ai_generate_module._constant(
                        self._series("a", "b"), name
                    )
                self.assertIn(name, str(caught.exception))

    def test_a_null_counts_as_a_distinct_value(self):
        # Otherwise half a column of nulls would silently take the other
        # half's setting.
        with self.assertRaises(ValueError):
            ai_generate_module._constant(self._series("a", None), "endpoint")


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
        self.assertEqual(row["status"], "")
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
                self.assertEqual(row["full_response"], "{}")
                self.assertEqual(row["status"], "")

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
                self.assertTrue(row["status"].startswith("SAFETY_BLOCKED:"))
                self.assertIn(reason.name, row["status"])
                # The reason is still recoverable from the full response.
                self.assertIn("candidates", row["full_response"])

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
        self.assertEqual(row["status"], "")
        self.assertEqual(row["result"], "hi")

    def test_a_failure_leaves_an_empty_json_document(self):
        # Never null, so that parse_json(full_response) is safe on every row.
        with mock.patch.object(ai_generate_module, "_MAX_ATTEMPTS", 1):
            row = generate_one(
                generate_content=replies_with(
                    api_error(404, "NOT_FOUND", "no such model")
                ),
                prompt="hi",
            )
        self.assertEqual(row["full_response"], "{}")
        self.assertEqual(json.loads(row["full_response"]), {})

    def test_the_status_leads_with_the_canonical_code(self):
        for code, status in [
            (429, "RESOURCE_EXHAUSTED"),
            (404, "NOT_FOUND"),
            (403, "PERMISSION_DENIED"),
        ]:
            with self.subTest(status=status):
                with mock.patch.object(
                    ai_generate_module, "_MAX_ATTEMPTS", 1
                ):
                    row = generate_one(
                        generate_content=replies_with(
                            api_error(code, status, "the detail")
                        ),
                        prompt="hi",
                    )
                self.assertIsNone(row["result"])
                self.assertTrue(row["status"].startswith(f"{status}:"))
                self.assertIn("the detail", row["status"])

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
        self.assertEqual(row["status"], "")

    def test_a_bug_in_this_library_is_not_hidden_in_status(self):
        async def _raise(**_):
            raise TypeError("this is a bug, not a backend failure")

        with self.assertRaises(TypeError):
            generate_one(generate_content=_raise, prompt="hi")

    def test_a_non_string_prompt_is_rendered(self):
        # Arrow delivers a nullable BIGINT column as floats, so a whole number
        # must not pick up a fraction just because another row was null.
        row = generate_one(generate_content=answers_the_prompt(), prompt=2.0)
        self.assertEqual(row["result"], "answer to 2")

    def test_a_malformed_part_is_reported_per_row(self):
        # Unlike a malformed setting, a bad URI is a property of one row.
        row = generate_one(
            generate_content=answers_the_prompt(),
            prompt=[{"uri": "gs://bucket/file-without-an-extension"}],
        )
        self.assertIsNone(row["result"])
        self.assertTrue(row["status"].startswith("INVALID_ARGUMENT:"))

    def test_a_prompt_of_only_null_parts_is_a_null_prompt(self):
        def _fail(**_):
            raise AssertionError("the model must not be called")

        row = generate_one(
            generate_content=_fail, prompt=[None, {"uri": None}]
        )
        self.assertIsNone(row["result"])
        self.assertEqual(row["status"], "")

    def test_parts_are_sent_in_order(self):
        row = generate_one(
            generate_content=echoes_the_parts(),
            prompt=["Read this: ", {"uri": "gs://b/x.pdf"}],
        )
        self.assertEqual(
            json.loads(row["result"]),
            [
                {"text": "Read this: "},
                {"uri": "gs://b/x.pdf", "mime_type": "application/pdf"},
            ],
        )


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
            self.assertEqual(row["status"], "")

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
                .select("g.result", "g.status", "g.full_response")
                .collect()
            )

        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertIsNone(row["result"])
            self.assertEqual(row["full_response"], "{}")

    def test_the_dead_letter_filter_selects_the_failed_rows(self):
        # The point of the empty status: one predicate finds what went wrong.
        with mock.patch.object(ai_generate_module, "_MAX_ATTEMPTS", 1):
            failed = (
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
                .where("g.status <> ''")
                .count()
            )
        # The null-prompt row never calls the model, so it succeeds.
        self.assertEqual(failed, 3)

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
        self.assertEqual(request["model"], ai_generate_module.DEFAULT_ENDPOINT)
        self.assertIsNotNone(request["response_schema"])

    def test_a_bad_setting_from_sql_fails_the_query(self):
        # A mistake in the query is not a property of a row: it must not cost
        # a full scan producing a column of failures.
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=answers_the_prompt()),
        )
        self._documents().createOrReplaceTempView("documents")

        with self.assertRaises(Exception) as caught:
            self.spark.sql(
                "SELECT ai_generate(prompt => body,"
                " output_schema => 'a NOTATYPE').status AS s FROM documents"
            ).collect()
        self.assertIn("NOTATYPE", str(caught.exception))

    def test_a_setting_that_varies_per_row_fails_the_query(self):
        # endpoint configures the call, not the row. BigQuery rejects a column
        # outright; Spark SQL cannot, so the value is checked as it arrives.
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=echoes_the_request()),
        )
        self.spark.createDataFrame(
            [("a", "gemini-2.5-pro"), ("b", "gemini-3.6-flash")],
            "body STRING, model STRING",
        ).repartition(1).createOrReplaceTempView("varying")

        with self.assertRaises(Exception) as caught:
            self.spark.sql(
                "SELECT ai_generate(prompt => body, endpoint => model).result"
                " AS r FROM varying"
            ).collect()
        self.assertIn("endpoint", str(caught.exception))

    def test_a_repeated_column_value_is_accepted(self):
        # A literal and a column that happens to repeat are indistinguishable
        # on the executor, so the constant one has to keep working.
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=echoes_the_request()),
        )
        self.spark.createDataFrame(
            [("a", "gemini-2.5-pro"), ("b", "gemini-2.5-pro")],
            "body STRING, model STRING",
        ).createOrReplaceTempView("constant")

        rows = self.spark.sql(
            "SELECT ai_generate(prompt => body, endpoint => model).result AS r"
            " FROM constant"
        ).collect()

        self.assertEqual(
            [json.loads(r["r"])["model"] for r in rows],
            ["gemini-2.5-pro", "gemini-2.5-pro"],
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


class TestMultimodalPrompts(SparkTestCase):
    """Struct prompts, through a real Spark plan in both dialects."""

    def _files(self):
        return self.spark.createDataFrame(
            [
                ("gs://bucket/invoice.pdf", "application/pdf"),
                ("gs://bucket/photo.png", None),
            ],
            "uri STRING, content_type STRING",
        )

    def test_a_file_only_prompt(self):
        rows = (
            self._files()
            .withColumn(
                "g",
                ai_generate(
                    file(sf.col("uri")),
                    _generate_content=echoes_the_parts(),
                ),
            )
            .select("uri", "g.result")
            .orderBy("uri")
            .collect()
        )

        self.assertEqual(
            [json.loads(row["result"]) for row in rows],
            [
                [
                    {
                        "uri": "gs://bucket/invoice.pdf",
                        "mime_type": "application/pdf",
                    }
                ],
                [{"uri": "gs://bucket/photo.png", "mime_type": "image/png"}],
            ],
        )

    def test_text_and_a_file_keep_their_order(self):
        rows = (
            self._files()
            .limit(1)
            .withColumn(
                "g",
                ai_generate(
                    [
                        sf.lit("Total? "),
                        file(sf.col("uri")),
                        sf.lit(" Answer in digits."),
                    ],
                    _generate_content=echoes_the_parts(),
                ),
            )
            .select("g.result")
            .collect()
        )

        self.assertEqual(
            json.loads(rows[0]["result"]),
            [
                {"text": "Total? "},
                {
                    "uri": "gs://bucket/invoice.pdf",
                    "mime_type": "application/pdf",
                },
                {"text": " Answer in digits."},
            ],
        )

    def test_the_content_type_column_overrides_detection(self):
        rows = (
            self._files()
            .limit(1)
            .withColumn(
                "g",
                ai_generate(
                    file(sf.col("uri"), sf.lit("text/plain")),
                    _generate_content=echoes_the_parts(),
                ),
            )
            .select("g.result")
            .collect()
        )

        self.assertEqual(
            json.loads(rows[0]["result"])[0]["mime_type"], "text/plain"
        )

    def test_a_named_struct_is_a_file_in_spark_sql(self):
        self.spark.udf.register(
            "ai_generate", ai_generate_udf(_generate_content=echoes_the_parts())
        )
        self._files().createOrReplaceTempView("files")

        rows = self.spark.sql(
            "SELECT ai_generate(named_struct('uri', uri)).result AS r"
            " FROM files ORDER BY uri"
        ).collect()

        self.assertEqual(
            [json.loads(row["r"])[0]["uri"] for row in rows],
            ["gs://bucket/invoice.pdf", "gs://bucket/photo.png"],
        )

    def test_a_struct_of_parts_in_spark_sql(self):
        self.spark.udf.register(
            "ai_generate", ai_generate_udf(_generate_content=echoes_the_parts())
        )
        self._files().createOrReplaceTempView("files")

        rows = self.spark.sql(
            "SELECT ai_generate("
            "  prompt => struct('Summarize: ', named_struct('uri', uri))"
            ").result AS r FROM files ORDER BY uri"
        ).collect()

        self.assertEqual(
            json.loads(rows[0]["r"]),
            [
                {"text": "Summarize: "},
                {
                    "uri": "gs://bucket/invoice.pdf",
                    "mime_type": "application/pdf",
                },
            ],
        )

    def test_an_unreadable_extension_is_reported_per_row(self):
        self.spark.udf.register(
            "ai_generate", ai_generate_udf(_generate_content=echoes_the_parts())
        )
        self.spark.createDataFrame(
            [("gs://bucket/invoice.pdf",), ("gs://bucket/mystery",)],
            "uri STRING",
        ).createOrReplaceTempView("mixed")

        rows = self.spark.sql(
            "SELECT uri, ai_generate(named_struct('uri', uri)) AS g"
            " FROM mixed ORDER BY uri"
        ).collect()

        # The good row still succeeds: only the row with the unusable name
        # reports a problem.
        self.assertEqual(rows[0]["g"]["status"], "")
        self.assertTrue(
            rows[1]["g"]["status"].startswith("INVALID_ARGUMENT:")
        )

    def test_an_unknown_field_is_rejected(self):
        self.spark.udf.register(
            "ai_generate", ai_generate_udf(_generate_content=echoes_the_parts())
        )
        self._files().createOrReplaceTempView("files")

        rows = self.spark.sql(
            "SELECT ai_generate(struct("
            "  named_struct('uri', uri, 'size', 1)"
            ")).status AS s FROM files"
        ).collect()

        for row in rows:
            self.assertIn("uri", row["s"])


if __name__ == "__main__":
    unittest.main()
