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
"""

import json
import unittest

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

from google.cloud.dataproc_ml.sql import ai_generate, ai_generate_udf

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

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

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
                self.assertEqual(row["status"], "SUCCESS")
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

    def test_full_response_carries_usage_metadata(self):
        result = self.cities.limit(1).withColumn(
            "g", ai_generate(self._prompt())
        )
        payload = json.loads(result.select("g.full_response").first()[0])

        self.assertIn("candidates", payload)
        self.assertIn("usage_metadata", payload)

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
            self.assertEqual(row["status"], "SUCCESS")
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
            self.assertEqual(row["status"], "SUCCESS")
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
                self.assertEqual(row["status"], "SUCCESS")
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
        from google.cloud.dataproc_ml.sql import _client

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

        self.assertEqual(rows[0]["status"], "SUCCESS")
        self.assertIsNotNone(rows[0]["result"])

    def test_unknown_model_is_reported_per_row(self):
        # A bad endpoint must not fail the query: every row reports the error
        # in its status.
        rows = (
            self.cities.limit(2)
            .withColumn(
                "g",
                ai_generate(
                    self._prompt(), endpoint="gemini-does-not-exist"
                ),
            )
            .select("g.result", "g.status")
            .collect()
        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNone(row["result"])
            self.assertIn("NOT_FOUND", row["status"])

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

    def test_spark_sql_with_named_arguments(self):
        # The call style the spec uses, end to end against the real model.
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.cities.createOrReplaceTempView("cities")

        rows = self.spark.sql(
            "SELECT city, expected, ai_generate("
            "  prompt => CONCAT('Airport code of the largest airport in ',"
            "                   city, '? Three letters only.'),"
            "  endpoint => 'gemini-3.6-flash',"
            f"  model_params => '{_DETERMINISTIC}',"
            "  output_schema => 'code STRING'"
            ") AS g FROM cities"
        ).collect()

        self.assertEqual(len(rows), 7)
        for row in rows:
            with self.subTest(city=row["city"]):
                self.assertEqual(row["g"]["status"], "SUCCESS")
                parsed = json.loads(row["g"]["result"])
                self.assertIn(row["expected"], parsed["code"].upper())

    def test_spark_sql_argument_may_vary_per_row(self):
        self.spark.udf.register("ai_generate", ai_generate_udf())
        self.spark.createDataFrame(
            [
                ("France", "Answer with the capital city only."),
                ("Japan", "Answer with the capital city only."),
            ],
            ["country", "instruction"],
        ).createOrReplaceTempView("questions")

        rows = self.spark.sql(
            "SELECT country, ai_generate("
            "  prompt => CONCAT(instruction, ' Country: ', country),"
            "  output_schema => 'capital STRING'"
            ").result AS r FROM questions ORDER BY country"
        ).collect()

        capitals = [json.loads(row["r"])["capital"] for row in rows]
        self.assertIn("Paris", capitals[0])
        self.assertIn("Tokyo", capitals[1])


if __name__ == "__main__":
    unittest.main()
