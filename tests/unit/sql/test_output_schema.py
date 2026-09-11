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

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from google.cloud.dataproc_ml.sql import _output_schema


class OutputSchemaTestCase(unittest.TestCase):
    """Parsing a schema string uses Spark's parser, so a session is needed."""

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("dataproc-ml-sql-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    def field_names(self, struct):
        return [field.name for field in struct.fields]

    def field_type(self, struct, name):
        return struct[name].dataType


class TestParsing(OutputSchemaTestCase):

    def test_bigquery_type_names_are_accepted(self):
        struct = _output_schema.parse_output_schema(
            "a STRING, b INT64, c FLOAT64, d BOOL"
        )
        self.assertIsInstance(self.field_type(struct, "a"), StringType)
        self.assertIsInstance(self.field_type(struct, "b"), LongType)
        self.assertIsInstance(self.field_type(struct, "c"), DoubleType)
        self.assertIsInstance(self.field_type(struct, "d"), BooleanType)

    def test_spark_type_names_are_accepted(self):
        struct = _output_schema.parse_output_schema(
            "a STRING, b BIGINT, c DOUBLE, d BOOLEAN"
        )
        self.assertIsInstance(self.field_type(struct, "b"), LongType)
        self.assertIsInstance(self.field_type(struct, "c"), DoubleType)
        self.assertIsInstance(self.field_type(struct, "d"), BooleanType)

    def test_a_field_named_after_a_type_keeps_its_name(self):
        # Rewriting BigQuery type names must not touch the field names, or
        # "int64 INT64" would silently declare a field called BIGINT.
        struct = _output_schema.parse_output_schema(
            "int64 INT64, bool BOOL, float64 FLOAT64"
        )
        self.assertEqual(self.field_names(struct), ["bool", "float64", "int64"])
        self.assertIsInstance(self.field_type(struct, "int64"), LongType)
        self.assertIsInstance(self.field_type(struct, "bool"), BooleanType)
        self.assertIsInstance(self.field_type(struct, "float64"), DoubleType)

    def test_a_type_keeps_its_modifiers(self):
        for schema in (
            "a INT64 NOT NULL",
            "a INT64 COMMENT 'how many'",
            "a INT64 NOT NULL, b BOOL",
        ):
            with self.subTest(schema=schema):
                struct = _output_schema.parse_output_schema(schema)
                self.assertIsInstance(self.field_type(struct, "a"), LongType)

    def test_arrays_and_structs_are_supported(self):
        struct = _output_schema.parse_output_schema(
            "tags ARRAY<STRING>, location STRUCT<city: STRING, zip: INT64>"
        )
        self.assertIsInstance(self.field_type(struct, "tags"), ArrayType)
        nested = self.field_type(struct, "location")
        self.assertEqual(self.field_names(nested), ["city", "zip"])
        self.assertIsInstance(nested["zip"].dataType, LongType)

    def test_a_struct_type_can_be_passed_directly(self):
        struct = _output_schema.parse_output_schema(
            StructType(
                [
                    StructField("b", StringType()),
                    StructField("a", LongType()),
                ]
            )
        )
        self.assertEqual(self.field_names(struct), ["a", "b"])

    def test_unparsable_schema_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _output_schema.parse_output_schema("this is not a schema")
        self.assertIn("Failed to parse", str(ctx.exception))

    def test_empty_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            _output_schema.parse_output_schema(StructType([]))

    def test_wrong_argument_type_is_rejected(self):
        with self.assertRaises(ValueError):
            _output_schema.parse_output_schema(42)


class TestFieldOrdering(OutputSchemaTestCase):
    """BigQuery returns custom schema fields sorted by name."""

    def test_fields_are_sorted_by_name(self):
        struct = _output_schema.parse_output_schema(
            "sentiment STRING, explanation STRING, score INT64"
        )
        self.assertEqual(
            self.field_names(struct), ["explanation", "score", "sentiment"]
        )

    def test_nested_struct_fields_are_sorted_too(self):
        struct = _output_schema.parse_output_schema(
            "location STRUCT<zip: INT64, city: STRING>"
        )
        self.assertEqual(
            self.field_names(self.field_type(struct, "location")),
            ["city", "zip"],
        )

    def test_struct_inside_an_array_is_sorted(self):
        struct = _output_schema.parse_output_schema(
            "items ARRAY<STRUCT<name: STRING, id: INT64>>"
        )
        element = self.field_type(struct, "items").elementType
        self.assertEqual(self.field_names(element), ["id", "name"])


class TestValidation(OutputSchemaTestCase):

    def test_unsupported_types_are_rejected(self):
        for declaration in ("d DATE", "t TIMESTAMP", "m MAP<STRING, STRING>"):
            with self.subTest(declaration=declaration):
                with self.assertRaises(ValueError) as ctx:
                    _output_schema.parse_output_schema(declaration)
                self.assertIn("Unsupported field type", str(ctx.exception))

    def test_options_clause_is_reported_as_unsupported(self):
        with self.assertRaises(ValueError) as ctx:
            _output_schema.parse_output_schema(
                "state STRING OPTIONS(description = 'the state')"
            )
        self.assertIn("OPTIONS", str(ctx.exception))

    def test_duplicate_field_names_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _output_schema.parse_output_schema(
                StructType(
                    [
                        StructField("a", StringType()),
                        StructField("a", StringType()),
                    ]
                )
            )
        self.assertIn("duplicate field names", str(ctx.exception))

    def test_deep_nesting_is_rejected(self):
        deep = "a STRUCT<b: STRUCT<c: STRUCT<d: STRUCT<e: STRING>>>>"
        with self.assertRaises(ValueError) as ctx:
            _output_schema.parse_output_schema(deep)
        self.assertIn("too many nested levels", str(ctx.exception))

    def test_moderate_nesting_is_allowed(self):
        struct = _output_schema.parse_output_schema(
            "a STRUCT<b: STRUCT<c: STRING>>"
        )
        self.assertEqual(self.field_names(struct), ["a"])


class TestResponseSchema(OutputSchemaTestCase):

    def test_scalars_map_to_gemini_types(self):
        struct = _output_schema.parse_output_schema(
            "a STRING, b INT64, c FLOAT64, d BOOL"
        )
        schema = _output_schema.to_response_schema(struct)
        self.assertEqual(schema["type"], "OBJECT")
        self.assertEqual(
            {
                name: value["type"]
                for name, value in schema["properties"].items()
            },
            {"a": "STRING", "b": "INTEGER", "c": "NUMBER", "d": "BOOLEAN"},
        )

    def test_arrays_declare_their_item_type(self):
        struct = _output_schema.parse_output_schema("tags ARRAY<STRING>")
        schema = _output_schema.to_response_schema(struct)
        self.assertEqual(
            schema["properties"]["tags"],
            {"type": "ARRAY", "items": {"type": "STRING"}},
        )

    def test_nested_structs_become_nested_objects(self):
        struct = _output_schema.parse_output_schema(
            "location STRUCT<city: STRING>"
        )
        schema = _output_schema.to_response_schema(struct)
        self.assertEqual(
            schema["properties"]["location"],
            {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}},
        )

    def test_non_nullable_fields_are_required(self):
        struct = _output_schema.parse_output_schema(
            StructType(
                [
                    StructField("a", StringType(), nullable=False),
                    StructField("b", StringType(), nullable=True),
                ]
            )
        )
        schema = _output_schema.to_response_schema(struct)
        self.assertEqual(schema["required"], ["a"])


if __name__ == "__main__":
    unittest.main()
