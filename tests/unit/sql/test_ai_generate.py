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

import asyncio
import json
import unittest
from unittest import mock

from google.genai import errors, types
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import LongType, StringType, StructField, StructType

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


def empty_response():
    """Builds a response with no usable text, as a filtered answer looks."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[]),
                finish_reason=types.FinishReason.SAFETY,
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

    def test_default_fields_match_bigquery(self):
        struct = ai_generate_module._build_return_type(None)
        self.assertEqual(
            [field.name for field in struct.fields],
            ["result", "full_response", "status"],
        )
        self.assertIsInstance(struct["result"].dataType, StringType)
        self.assertIsInstance(struct["status"].dataType, StringType)

    def test_output_schema_replaces_the_result_field(self):
        struct = ai_generate_module._build_return_type(
            StructType(
                [
                    StructField("sentiment", StringType()),
                    StructField("score", LongType()),
                ]
            )
        )
        self.assertEqual(
            [field.name for field in struct.fields],
            ["sentiment", "score", "full_response", "status"],
        )

    def test_reserved_field_names_are_rejected(self):
        for reserved in ("full_response", "status"):
            with self.subTest(field=reserved):
                with self.assertRaises(ValueError):
                    ai_generate_module._build_return_type(
                        StructType([StructField(reserved, StringType())])
                    )

    def test_generated_fields_are_nullable(self):
        # A failed row reports the failure in status and leaves the generated
        # fields empty, so they must accept nulls.
        struct = ai_generate_module._build_return_type(
            StructType([StructField("a", StringType(), nullable=False)])
        )
        self.assertTrue(struct["a"].nullable)


class TestValidation(SparkTestCase):
    """Configuration mistakes are reported on the driver, before any job."""

    def test_empty_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            ai_generate_udf(endpoint="  ")

    def test_invalid_model_params_are_rejected(self):
        with self.assertRaises(ValueError):
            ai_generate_udf(model_params={"contents": ["hi"]})


class TestGenerateOneRow(SparkTestCase):
    """The per row behavior, exercised without Spark for precision."""

    def _run(
        self,
        generate_content,
        prompt="a prompt",
        *,
        max_attempts=5,
        **overrides,
    ):
        options = {
            "generate_content": generate_content,
            "prompt": prompt,
            "endpoint": "gemini-3.6-flash",
            "config": None,
            "schema": None,
        }
        options.update(overrides)
        # The retry settings are module constants. They are patched here so
        # that the tests neither sleep nor depend on the shipped values.
        with mock.patch.multiple(
            ai_generate_module,
            _MAX_ATTEMPTS=max_attempts,
            _BASE_RETRY_DELAY_SECONDS=0.0,
            _MAX_RETRY_DELAY_SECONDS=0.0,
        ):
            return asyncio.run(ai_generate_module._generate_one(**options))

    def test_successful_row_has_empty_status(self):
        row = self._run(replies_with(response_with_text("BLR")))
        self.assertEqual(row["result"], "BLR")
        self.assertEqual(row["status"], "")

    def test_reasoning_parts_are_skipped(self):
        row = self._run(
            replies_with(response_with_text("BLR", thought="thinking..."))
        )
        self.assertEqual(row["result"], "BLR")

    def test_response_without_text_produces_no_result(self):
        # A filtered answer is not an error, so status stays empty.
        row = self._run(replies_with(empty_response()))
        self.assertIsNone(row["result"])
        self.assertEqual(row["status"], "")

    def test_null_prompt_skips_the_model(self):
        called = []

        async def _never(*, contents, **_kwargs):
            called.append(contents)
            raise AssertionError("the model must not be called")

        for prompt in (None, float("nan")):
            with self.subTest(prompt=prompt):
                row = self._run(_never, prompt=prompt)
                self.assertIsNone(row["result"])
                self.assertEqual(row["status"], "")
        self.assertEqual(called, [])

    def test_transient_errors_are_retried_then_succeed(self):
        row = self._run(
            replies_with(
                api_error(429, "RESOURCE_EXHAUSTED", "slow down"),
                api_error(503, "UNAVAILABLE", "backend down"),
                response_with_text("BLR"),
            )
        )
        self.assertEqual(row["result"], "BLR")
        self.assertEqual(row["status"], "")

    def test_exhausted_retries_report_the_error(self):
        row = self._run(
            replies_with(api_error(429, "RESOURCE_EXHAUSTED", "slow down")),
            max_attempts=3,
        )
        self.assertIsNone(row["result"])
        self.assertEqual(
            row["status"],
            "A retryable error occurred: RESOURCE_EXHAUSTED error from remote "
            "service/endpoint.",
        )

    def test_request_errors_are_not_retried(self):
        attempts = []

        async def _fail(*, model, contents, config):
            del model, contents, config
            attempts.append(1)
            raise api_error(404, "NOT_FOUND", "Publisher Model was not found")

        row = self._run(_fail)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(
            row["status"], "NOT_FOUND: Publisher Model was not found"
        )

    def test_library_errors_are_not_hidden_in_status(self):
        async def _bug(*, model, contents, config):
            del model, contents, config
            raise KeyError("a bug in the library")

        with self.assertRaises(KeyError):
            self._run(_bug)

    def test_structured_output_is_mapped_to_fields(self):
        schema = StructType(
            [
                StructField("score", LongType()),
                StructField("sentiment", StringType()),
            ]
        )
        row = self._run(
            replies_with(
                response_with_text(
                    json.dumps({"sentiment": "positive", "score": 5})
                )
            ),
            schema=schema,
        )
        self.assertEqual(row["sentiment"], "positive")
        self.assertEqual(row["score"], 5)
        self.assertEqual(row["status"], "")
        self.assertNotIn("result", row)

    def test_unparsable_structured_output_is_reported(self):
        schema = StructType([StructField("sentiment", StringType())])
        row = self._run(
            replies_with(response_with_text("not json at all")),
            schema=schema,
        )
        self.assertIsNone(row["sentiment"])
        self.assertEqual(
            row["status"],
            "Failed to parse the response - please check the full_response "
            "field.",
        )
        self.assertIsNotNone(row["full_response"])

    def test_missing_fields_become_null(self):
        schema = StructType(
            [
                StructField("sentiment", StringType()),
                StructField("score", LongType()),
            ]
        )
        row = self._run(
            replies_with(
                response_with_text(json.dumps({"sentiment": "positive"}))
            ),
            schema=schema,
        )
        self.assertEqual(row["sentiment"], "positive")
        self.assertIsNone(row["score"])


class TestSparkExecution(SparkTestCase):
    """The function as users see it, running through real Spark tasks."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.df = cls.spark.createDataFrame(
            [("Bengaluru",), ("London",), ("Tokyo",), ("Paris",), ("Oslo",)],
            ["city"],
        ).repartition(2)

    def test_default_schema_and_dot_access(self):
        result = self.df.withColumn(
            "g",
            ai_generate(sf.col("city"), _generate_content=answers_the_prompt()),
        )
        self.assertEqual(
            [field.name for field in result.schema["g"].dataType.fields],
            ["result", "full_response", "status"],
        )

        rows = result.select("city", "g.result", "g.status").collect()
        self.assertEqual(len(rows), 5)
        for row in rows:
            city = row["city"]
            self.assertEqual(row["result"], f"answer to {city}")
            self.assertEqual(row["status"], "")

    def test_prompt_parts_reach_the_model(self):
        result = self.df.withColumn(
            "g",
            ai_generate(
                [sf.lit("Airport code for "), sf.col("city")],
                _generate_content=answers_the_prompt(),
            ),
        )
        rows = result.select("city", "g.result").collect()
        for row in rows:
            city = row["city"]
            self.assertEqual(
                row["result"], f"answer to Airport code for {city}"
            )

    def test_full_response_holds_the_whole_response(self):
        result = self.df.withColumn(
            "g",
            ai_generate(sf.col("city"), _generate_content=answers_the_prompt()),
        )
        # The column is VARIANT on Spark 4 and a JSON string otherwise, so it
        # is read back as text either way.
        as_text = result.selectExpr(
            "CAST(to_json(g.full_response) AS STRING) AS payload"
            if ai_generate_module._VARIANT_SUPPORTED
            else "g.full_response AS payload"
        ).first()["payload"]
        payload = json.loads(as_text)
        self.assertIn("candidates", payload)

    def test_null_rows_produce_null_results(self):
        df = self.spark.createDataFrame([("Oslo",), (None,)], ["city"])
        rows = (
            df.withColumn(
                "g",
                ai_generate(
                    sf.col("city"), _generate_content=answers_the_prompt()
                ),
            )
            .select("city", "g.result", "g.status")
            .collect()
        )
        by_city = {row["city"]: row for row in rows}
        self.assertEqual(by_city["Oslo"]["result"], "answer to Oslo")
        self.assertIsNone(by_city[None]["result"])
        self.assertEqual(by_city[None]["status"], "")

    def test_structured_output_columns(self):
        async def _structured(*, model, contents, config):
            del model, config
            return response_with_text(
                json.dumps({"city": contents, "letters": len(contents)})
            )

        result = self.df.withColumn(
            "g",
            ai_generate(
                sf.col("city"),
                output_schema="letters INT64, city STRING",
                _generate_content=_structured,
            ),
        )
        self.assertEqual(
            [field.name for field in result.schema["g"].dataType.fields],
            ["city", "letters", "full_response", "status"],
        )

        rows = result.select("g.city", "g.letters").collect()
        for row in rows:
            self.assertEqual(row["letters"], len(row["city"]))

    def test_registered_for_spark_sql(self):
        self.spark.udf.register(
            "ai_generate",
            ai_generate_udf(_generate_content=answers_the_prompt()),
        )
        self.df.createOrReplaceTempView("cities")

        rows = self.spark.sql(
            "SELECT city, ai_generate(city).result AS answer,"
            " ai_generate(city).status AS status FROM cities"
        ).collect()

        self.assertEqual(len(rows), 5)
        for row in rows:
            city = row["city"]
            self.assertEqual(row["answer"], f"answer to {city}")
            self.assertEqual(row["status"], "")

    def test_failed_rows_do_not_fail_the_query(self):
        result = self.df.withColumn(
            "g",
            ai_generate(
                sf.col("city"),
                _generate_content=replies_with(
                    api_error(404, "NOT_FOUND", "Publisher Model not found")
                ),
            ),
        )

        rows = result.select("g.result", "g.status").collect()
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertIsNone(row["result"])
            self.assertEqual(
                row["status"], "NOT_FOUND: Publisher Model not found"
            )


if __name__ == "__main__":
    unittest.main()
