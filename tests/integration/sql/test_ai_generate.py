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
from google.cloud.dataproc_ml.sql import _ai_generate as ai_generate_module

_AIRPORT_PROMPT = (
    "What is the airport code of the largest airport in {city}? "
    "Answer with the three letter code only."
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
                self.assertEqual(row["status"], "")
                self.assertIsNotNone(row["result"])
                self.assertIn(row["expected"], row["result"].upper())

    def test_full_response_carries_usage_metadata(self):
        result = self.cities.limit(1).withColumn(
            "g", ai_generate(self._prompt())
        )
        payload_column = (
            "to_json(g.full_response)"
            if ai_generate_module._VARIANT_SUPPORTED
            else "g.full_response"
        )
        payload = json.loads(
            result.selectExpr(f"{payload_column} AS payload").first()["payload"]
        )

        self.assertIn("candidates", payload)
        self.assertIn("usage_metadata", payload)

    def test_model_params_are_applied(self):
        result = (
            self.cities.limit(2)
            .withColumn(
                "g",
                ai_generate(
                    self._prompt(),
                    model_params={
                        "generation_config": {
                            "temperature": 0.0,
                            "thinking_config": {"thinking_budget": 0},
                        }
                    },
                ),
            )
            .select("g.result", "g.status")
            .collect()
        )

        for row in result:
            self.assertEqual(row["status"], "")
            self.assertIsNotNone(row["result"])

    def test_structured_output(self):
        reviews = self.spark.createDataFrame(
            [
                ("The battery lasts two days and the screen is gorgeous.",),
                ("It broke after a week and support never replied.",),
            ],
            ["review"],
        )

        result = reviews.withColumn(
            "g",
            ai_generate(
                [
                    sf.lit("Analyze this product review: "),
                    sf.col("review"),
                ],
                output_schema=(
                    "sentiment STRING, score INT64, topics ARRAY<STRING>"
                ),
                model_params={"generation_config": {"temperature": 0.0}},
            ),
        )

        # BigQuery returns custom schema fields sorted by name.
        self.assertEqual(
            [field.name for field in result.schema["g"].dataType.fields],
            ["score", "sentiment", "topics", "full_response", "status"],
        )

        rows = result.select(
            "g.sentiment", "g.score", "g.topics", "g.status"
        ).collect()
        self.assertEqual(len(rows), 2)
        for row in rows:
            with self.subTest(sentiment=row["sentiment"]):
                self.assertEqual(row["status"], "")
                self.assertIsInstance(row["sentiment"], str)
                self.assertIsInstance(row["score"], int)
                self.assertTrue(row["topics"])

    def test_unknown_model_is_reported_per_row(self):
        # A bad endpoint must not fail the query: every row reports the error
        # in its status, exactly as BigQuery's AI.GENERATE does.
        rows = (
            self.cities.limit(2)
            .withColumn(
                "g",
                ai_generate(
                    self._prompt(),
                    endpoint="gemini-does-not-exist",
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


if __name__ == "__main__":
    unittest.main()
