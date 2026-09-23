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

"""Integration tests that call Vertex AI for real.

They rely on Application Default Credentials and, unless GOOGLE_CLOUD_PROJECT
is set, on the project discovered from the environment. The location is not
taken from the environment: these tests exercise the default global endpoint.
Running them incurs Vertex AI charges.

The multimodal tests read fixtures from a public-to-the-project bucket; the
files are read by Vertex AI, not by Spark, so the caller's credentials are what
must be able to see them.
"""

import json
import unittest

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from google.cloud.dataproc_ml.sql import _client
from google.cloud.dataproc_ml.sql import ai_generate, ai_generate_udf, file
from tests.utils.spark_version import requires_spark_4

_AIRPORT_PROMPT = (
    "What is the airport code of the largest airport in {city}? "
    "Answer with the three letter code only."
)

# Keeping the model from thinking or improvising makes the assertions about
# content stable without making them weaker.
_DETERMINISTIC = json.dumps(
    {
        "generation_config": {
            "temperature": 0.0,
            "thinking_config": {"thinking_budget": 0},
        }
    }
)

_FIXTURES = "gs://dataproc_ai_functions_test"
#: A product photo of a pet bed with the brand printed on a label.
_DOG_BED = f"{_FIXTURES}/images/playful-pup-dog-bed.png"
#: A close-up photograph of a white daisy. A JPEG, so that MIME detection is
#: exercised on more than one extension.
_FLOWER = f"{_FIXTURES}/images/100080576_f52e8ee070_n.jpg"
#: A one page invoice addressed to John Doe, number 001.
_INVOICE = f"{_FIXTURES}/pdfs/invoice.pdf"


class TestAiGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.cities = cls.spark.createDataFrame(
            [
                ("Bengaluru", "BLR"),
                ("London", "LHR"),
                ("San Francisco", "SFO"),
                ("Paris", "CDG"),
                ("Tokyo", "HND"),
                ("Sydney", "SYD"),
                ("New York", "JFK"),
            ],
            ["city", "expected"],
            # Several rows per batch, spread over more than one task.
        ).repartition(3)

    def _prompt(self):
        before, after = _AIRPORT_PROMPT.split("{city}", maxsplit=1)
        return [sf.lit(before), sf.col("city"), sf.lit(after)]

    def test_generates_an_answer_for_every_row(self):
        rows = (
            self.cities.withColumn("g", ai_generate(self._prompt()))
            .select("city", "expected", "g.result", "g.status")
            .collect()
        )

        self.assertEqual(len(rows), 7)
        for row in rows:
            with self.subTest(city=row["city"]):
                self.assertEqual(row["status"], "")
                self.assertIsNotNone(row["result"])
                self.assertIn(row["expected"], row["result"].upper())

    def test_the_return_type_is_always_the_same(self):
        plain = self.cities.withColumn("g", ai_generate(self._prompt()))
        structured = self.cities.withColumn(
            "g",
            ai_generate(self._prompt(), output_schema="code STRING"),
        )

        self.assertEqual(
            plain.schema["g"].dataType.simpleString(),
            "struct<result:string,full_response:string,status:string>",
        )
        self.assertEqual(
            plain.schema["g"].dataType, structured.schema["g"].dataType
        )

    def test_the_dead_letter_filter_finds_nothing_when_all_rows_succeed(self):
        # The contract that makes the empty status worth having: one predicate
        # separates the rows that need attention from the rest.
        result = self.cities.withColumn("g", ai_generate(self._prompt()))

        self.assertEqual(result.where("g.status <> ''").count(), 0)
        self.assertEqual(result.where("g.status = ''").count(), 7)

    def test_full_response_carries_usage_metadata(self):
        result = self.cities.limit(1).withColumn(
            "g", ai_generate(self._prompt())
        )
        payload = json.loads(result.select("g.full_response").first()[0])

        self.assertIn("candidates", payload)
        self.assertIn("usage_metadata", payload)

    @requires_spark_4
    def test_full_response_can_be_queried_with_parse_json(self):
        # The field is a string, so anything structural goes through
        # parse_json. This is the documented way to reach inside it.
        result = (
            self.cities.limit(1)
            .withColumn("g", ai_generate(self._prompt()))
            .selectExpr(
                "variant_get(parse_json(g.full_response),"
                " '$.usage_metadata.total_token_count', 'int') AS tokens"
            )
        )
        self.assertGreater(result.first()["tokens"], 0)

    def test_model_params_are_applied(self):
        rows = (
            self.cities.limit(2)
            .withColumn(
                "g",
                ai_generate(self._prompt(), model_params=_DETERMINISTIC),
            )
            .select("g.result", "g.status")
            .collect()
        )

        for row in rows:
            self.assertEqual(row["status"], "")
            self.assertIsNotNone(row["result"])

    def test_model_params_can_cap_the_output(self):
        # A visible, checkable effect, rather than merely "it did not fail".
        rows = (
            self.cities.limit(2)
            .withColumn(
                "g",
                ai_generate(
                    [sf.lit("Write 500 words about "), sf.col("city")],
                    model_params=json.dumps(
                        {
                            "generation_config": {
                                "max_output_tokens": 20,
                                "thinking_config": {"thinking_budget": 0},
                            }
                        }
                    ),
                ),
            )
            .select("g.result", "g.status")
            .collect()
        )

        for row in rows:
            self.assertEqual(row["status"], "")
            self.assertLess(len(row["result"].split()), 60)

    def test_structured_output_is_json_in_the_result_field(self):
        reviews = self.spark.createDataFrame(
            [
                ("The battery lasts two days and the screen is gorgeous.",),
                ("It broke after a week and support never replied.",),
            ],
            ["review"],
        )

        rows = (
            reviews.withColumn(
                "g",
                ai_generate(
                    [
                        sf.lit("Analyze this product review: "),
                        sf.col("review"),
                    ],
                    output_schema=(
                        "sentiment STRING, score INT, topics ARRAY<STRING>"
                    ),
                    model_params=_DETERMINISTIC,
                ),
            )
            .select("g.result", "g.status")
            .collect()
        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            with self.subTest(result=row["result"]):
                self.assertEqual(row["status"], "")
                # Constrained decoding guarantees this parses and conforms.
                parsed = json.loads(row["result"])
                self.assertIsInstance(parsed["sentiment"], str)
                self.assertIsInstance(parsed["score"], int)
                self.assertIsInstance(parsed["topics"], list)

    def test_structured_output_can_be_projected_with_from_json(self):
        # The caller chooses when to pay for parsing, and picks the Spark
        # types, rather than the function deciding the return type for them.
        schema = "sentiment STRING, score INT"
        rows = (
            self.spark.createDataFrame(
                [("Absolutely delightful, would buy again.",)], ["review"]
            )
            .withColumn(
                "g",
                ai_generate(
                    [sf.lit("Analyze: "), sf.col("review")],
                    output_schema=schema,
                    model_params=_DETERMINISTIC,
                ),
            )
            .select(sf.from_json(sf.col("g.result"), schema).alias("parsed"))
            .select("parsed.sentiment", "parsed.score")
            .collect()
        )

        self.assertEqual(len(rows), 1)
        self.assertIsInstance(rows[0]["sentiment"], str)
        self.assertIsInstance(rows[0]["score"], int)

    def test_a_fully_qualified_endpoint_is_accepted(self):
        project, _ = _client.resolve_project_and_location(None, None)
        self.assertIsNotNone(project, "the test project must be discoverable")

        path = (
            f"projects/{project}/locations/global/publishers/google/models/"
            "gemini-3.6-flash"
        )
        rows = (
            self.cities.limit(1)
            .withColumn("g", ai_generate(self._prompt(), endpoint=path))
            .select("g.result", "g.status")
            .collect()
        )

        self.assertEqual(rows[0]["status"], "")
        self.assertIsNotNone(rows[0]["result"])

    def test_an_unknown_model_fails_the_query(self):
        # A typo in the model name is a mistake in the query, not a property
        # of a row: it must not cost a full scan producing a column of
        # failures. BigQuery likewise rejects it before returning any row.
        with self.assertRaises(Exception) as caught:
            self.cities.withColumn(
                "g",
                ai_generate(self._prompt(), endpoint="gemini-does-not-exist"),
            ).collect()

        self.assertIn("gemini-does-not-exist", str(caught.exception))

    def test_registered_for_spark_sql(self):
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.cities.createOrReplaceTempView("cities")

        rows = self.spark.sql(
            "SELECT city,"
            " ai_generate(CONCAT('Airport code of the largest airport in ',"
            " city, '? Three letters only.')).result AS code"
            " FROM cities"
        ).collect()

        self.assertEqual(len(rows), 7)
        for row in rows:
            self.assertIsNotNone(row["code"])

    def test_spark_sql_with_options(self):
        # The call style the spec uses, end to end against the real model.
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.cities.createOrReplaceTempView("cities")

        rows = self.spark.sql(
            "SELECT city, expected, ai_generate("
            "  CONCAT('Airport code of the largest airport in ',"
            "         city, '? Three letters only.'),"
            "  'gemini-3.6-flash',"
            "  named_struct("
            f"    'model_params', '{_DETERMINISTIC}',"
            "     'output_schema', 'code STRING'"
            "  )"
            ") AS g FROM cities"
        ).collect()

        self.assertEqual(len(rows), 7)
        for row in rows:
            with self.subTest(city=row["city"]):
                self.assertEqual(row["g"]["status"], "")
                parsed = json.loads(row["g"]["result"])
                self.assertIn(row["expected"], parsed["code"].upper())

    def test_options_may_replace_endpoint(self):
        # endpoint is a string and options is a struct, so they are told apart
        # by type and a caller never has to write a NULL placeholder.
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.cities.limit(2).createOrReplaceTempView("few_cities")

        rows = self.spark.sql(
            "SELECT city, expected, ai_generate("
            "  CONCAT('Airport code of the largest airport in ',"
            "         city, '? Three letters only.'),"
            "  named_struct("
            f"    'model_params', '{_DETERMINISTIC}',"
            "     'output_schema', 'code STRING'"
            "  )"
            ") AS g FROM few_cities"
        ).collect()

        self.assertEqual(len(rows), 2)
        for row in rows:
            with self.subTest(city=row["city"]):
                self.assertEqual(row["g"]["status"], "")
                parsed = json.loads(row["g"]["result"])
                self.assertIn(row["expected"], parsed["code"].upper())

    def test_an_unknown_option_fails_the_query(self):
        # A typo must not look like a setting that quietly took effect.
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.cities.createOrReplaceTempView("cities")

        with self.assertRaises(Exception) as caught:
            self.spark.sql(
                "SELECT ai_generate(city,"
                " named_struct('output_schemaa', 'code STRING')).result AS r"
                " FROM cities"
            ).collect()

        self.assertIn("output_schemaa", str(caught.exception))

    def test_a_setting_that_varies_per_row_fails_the_query(self):
        # endpoint, model_params and output_schema configure the call, not the
        # row. BigQuery rejects a column for them at plan time; Spark SQL
        # cannot tell a literal from a column, so the value is checked as it
        # arrives and a second distinct one fails the query.
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.spark.createDataFrame(
            [
                ("France", "capital STRING"),
                ("Japan", "city STRING"),
            ],
            ["country", "schema"],
        ).repartition(1).createOrReplaceTempView("questions")

        with self.assertRaises(Exception) as caught:
            self.spark.sql(
                "SELECT ai_generate("
                "  CONCAT('Capital of ', country, '?'),"
                "  NULL,"
                "  named_struct('output_schema', schema)"
                ").result AS r FROM questions"
            ).collect()

        self.assertIn("output_schema", str(caught.exception))


class TestMultimodalPrompts(unittest.TestCase):
    """Prompts that include a file, against the real model."""

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.getOrCreate()
        cls.images = cls.spark.createDataFrame(
            [(_DOG_BED,), (_FLOWER,)], ["uri"]
        )

    def test_text_and_an_image(self):
        # The brand is printed on a label in the photo, so a correct answer
        # can only come from having actually read the image.
        rows = (
            self.spark.createDataFrame([(_DOG_BED,)], ["uri"])
            .withColumn(
                "g",
                ai_generate(
                    [
                        sf.lit(
                            "What brand name is printed on the label of this "
                            "product? Answer with the brand name only."
                        ),
                        file(sf.col("uri")),
                    ],
                    model_params=_DETERMINISTIC,
                ),
            )
            .select("g.result", "g.status")
            .collect()
        )

        self.assertEqual(rows[0]["status"], "")
        self.assertIn("playful pup", rows[0]["result"].lower())

    def test_a_file_on_its_own_is_a_prompt(self):
        # named_struct('uri', ...) with nothing else is one file, not a part
        # list holding one string.
        rows = (
            self.spark.createDataFrame([(_FLOWER,)], ["uri"])
            .withColumn(
                "g",
                ai_generate(
                    file(sf.col("uri")),
                    output_schema="subject STRING",
                    model_params=_DETERMINISTIC,
                ),
            )
            .select("g.result", "g.status")
            .collect()
        )

        self.assertEqual(rows[0]["status"], "")
        subject = json.loads(rows[0]["result"])["subject"].lower()
        self.assertTrue(
            any(word in subject for word in ("flower", "daisy", "petal")),
            subject,
        )

    def test_the_media_type_is_inferred_from_the_extension(self):
        # One PNG and one JPEG in the same column, with no content_type given.
        rows = (
            self.images.withColumn(
                "g",
                ai_generate(
                    [
                        sf.lit(
                            "Is the main subject of this image a plant or a "
                            "manufactured object? Answer 'plant' or 'object'."
                        ),
                        file(sf.col("uri")),
                    ],
                    model_params=_DETERMINISTIC,
                ),
            )
            .select("uri", "g.result", "g.status")
            .orderBy("uri")
            .collect()
        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["status"], "")
        # Ordered by URI: the numeric flower filename sorts before "playful".
        self.assertIn("plant", rows[0]["result"].lower())
        self.assertIn("object", rows[1]["result"].lower())

    def test_an_explicit_content_type_is_used(self):
        rows = (
            self.spark.createDataFrame([(_INVOICE, "application/pdf")],
                                       ["uri", "kind"])
            .withColumn(
                "g",
                ai_generate(
                    [
                        sf.lit("Who is this invoice addressed to?"),
                        file(sf.col("uri"), sf.col("kind")),
                    ],
                    model_params=_DETERMINISTIC,
                ),
            )
            .select("g.result", "g.status")
            .collect()
        )

        self.assertEqual(rows[0]["status"], "")
        self.assertIn("john doe", rows[0]["result"].lower())

    def test_a_pdf_with_structured_output(self):
        rows = (
            self.spark.createDataFrame([(_INVOICE,)], ["uri"])
            .withColumn(
                "g",
                ai_generate(
                    [
                        sf.lit("Extract the invoice details."),
                        file(sf.col("uri")),
                    ],
                    output_schema="invoice_number STRING, recipient STRING",
                    model_params=_DETERMINISTIC,
                ),
            )
            .select("g.result", "g.status")
            .collect()
        )

        self.assertEqual(rows[0]["status"], "")
        parsed = json.loads(rows[0]["result"])
        self.assertIn("001", parsed["invoice_number"])
        self.assertIn("john doe", parsed["recipient"].lower())

    def test_a_named_struct_is_a_file_in_spark_sql(self):
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.spark.createDataFrame(
            [(_DOG_BED,)], ["uri"]
        ).createOrReplaceTempView("documents")

        row = self.spark.sql(
            "SELECT ai_generate("
            "  struct("
            "    'What brand name is on the label? Brand name only.',"
            "    named_struct('uri', uri)"
            "  ),"
            f"  named_struct('model_params', '{_DETERMINISTIC}')"
            ") AS g FROM documents"
        ).first()

        self.assertEqual(row["g"]["status"], "")
        self.assertIn("playful pup", row["g"]["result"].lower())

    def test_a_content_type_field_in_spark_sql(self):
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.spark.createDataFrame(
            [(_INVOICE,)], ["uri"]
        ).createOrReplaceTempView("documents")

        row = self.spark.sql(
            "SELECT ai_generate("
            "  struct("
            "    'What is the invoice number? Digits only.',"
            "    named_struct('uri', uri, 'content_type', 'application/pdf')"
            "  ),"
            f"  named_struct('model_params', '{_DETERMINISTIC}')"
            ") AS g FROM documents"
        ).first()

        self.assertEqual(row["g"]["status"], "")
        self.assertIn("001", row["g"]["result"])

    def test_a_bad_uri_is_reported_per_row(self):
        # Unlike a bad setting, a bad file is a property of one row: the good
        # rows still produce answers.
        rows = (
            self.spark.createDataFrame(
                [(_FLOWER,), (f"{_FIXTURES}/images/no-such-image.png",)],
                ["uri"],
            )
            .withColumn(
                "g",
                ai_generate(
                    [sf.lit("Name the subject in one word."),
                     file(sf.col("uri"))],
                    model_params=_DETERMINISTIC,
                ),
            )
            .select("uri", "g.result", "g.status", "g.full_response")
            .orderBy("uri")
            .collect()
        )

        self.assertEqual(rows[0]["status"], "")
        self.assertIsNotNone(rows[0]["result"])
        self.assertNotEqual(rows[1]["status"], "")
        self.assertIsNone(rows[1]["result"])
        # Never null, so the column can always be given to parse_json.
        self.assertEqual(rows[1]["full_response"], "{}")

    def test_a_uri_with_no_usable_extension_is_reported_per_row(self):
        rows = (
            self.spark.createDataFrame(
                [(f"{_FIXTURES}/images/mystery",)], ["uri"]
            )
            .withColumn(
                "g",
                ai_generate(
                    [sf.lit("Describe this."), file(sf.col("uri"))],
                    model_params=_DETERMINISTIC,
                ),
            )
            .select("g.status")
            .collect()
        )

        self.assertTrue(rows[0]["status"].startswith("INVALID_ARGUMENT:"))
        self.assertIn("content_type", rows[0]["status"])


if __name__ == "__main__":
    unittest.main()
