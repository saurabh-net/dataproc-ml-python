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

"""Tests for the session-free ``output_schema`` DDL parser.

The parser exists because Spark's own ``StructType.fromDDL`` needs a session
and so cannot run on an executor. The risk in owning a second parser is that it
quietly drifts from the first, so most of this file does not assert the
expected output by hand: it asserts that the two parsers agree.
"""

import unittest

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructType,
)

from google.cloud.dataproc_ml.sql import _ddl
from tests.utils.spark_version import requires_spark_4

# The reference conversion, expressed against Spark's own parsed types. Keeping
# it in the test rather than in the library is deliberate: it is the oracle the
# parser is checked against, not a second implementation to maintain.
_REFERENCE_SCALARS = {
    StringType: "STRING",
    BooleanType: "BOOLEAN",
    ByteType: "INTEGER",
    ShortType: "INTEGER",
    IntegerType: "INTEGER",
    LongType: "INTEGER",
    FloatType: "NUMBER",
    DoubleType: "NUMBER",
}


def _reference_schema(data_type: DataType):
    """Converts a Spark type into the response schema the parser should emit."""
    if isinstance(data_type, StructType):
        schema = {
            "type": "OBJECT",
            "properties": {
                field.name: _reference_schema(field.dataType)
                for field in data_type.fields
            },
        }
        required = [
            field.name for field in data_type.fields if not field.nullable
        ]
        if required:
            schema["required"] = required
        return schema
    if isinstance(data_type, ArrayType):
        return {
            "type": "ARRAY",
            "items": _reference_schema(data_type.elementType),
        }
    return {"type": _REFERENCE_SCALARS[type(data_type)]}


# Every schema here must parse identically under both parsers.
_AGREEING_SCHEMAS = [
    # The spec's own examples.
    "sentiment STRING, urgency INT, categories ARRAY<STRING>",
    "summary STRING",
    # The full set of supported scalars, in both spellings where Spark has two.
    "a STRING, b BOOLEAN",
    "a BYTE, b TINYINT, c SHORT, d SMALLINT",
    "a INT, b INTEGER, c LONG, d BIGINT",
    "a FLOAT, b REAL, c DOUBLE",
    # Nesting.
    "tags ARRAY<STRING>",
    "matrix ARRAY<ARRAY<INT>>",
    "person STRUCT<name: STRING, age: INT>",
    "person STRUCT<name STRING, age INT>",
    "people ARRAY<STRUCT<name: STRING, pets ARRAY<STRING>>>",
    # Modifiers.
    "name STRING NOT NULL, age INT",
    "name STRING NOT NULL, age INT NOT NULL",
    "a STRING COMMENT 'why', b INT",
    "a STRING NOT NULL COMMENT 'why'",
    "person STRUCT<name: STRING NOT NULL, age: INT>",
    # Identifiers that need quoting, and fields named after types.
    "`odd name` STRING",
    "`select` STRING, `from` INT",
    "int STRING, string INT",
    # A schema wrapped in STRUCT<> rather than written as a bare field list.
    "STRUCT<a: STRING, b: ARRAY<INT>>",
    # Whitespace and casing.
    "  a   string ,   b   Int  ",
    "A STRING, b STRING",
]


class TestAgreesWithSpark(unittest.TestCase):
    """Pins the parser to Spark's, so the two cannot drift apart."""

    @classmethod
    def setUpClass(cls):
        # Only the oracle needs a session. The parser under test is called
        # without one on an executor, which is why it exists.
        cls.spark = SparkSession.builder.master("local[1]").getOrCreate()

    @requires_spark_4
    def test_matches_fromddl(self):
        # StructType.fromDDL is the oracle and was added in PySpark 4, so on
        # 3.5 there is nothing to compare against. The parser itself is
        # version independent and its other tests still run.
        for schema in _AGREEING_SCHEMAS:
            with self.subTest(schema=schema):
                expected = _reference_schema(StructType.fromDDL(schema))
                self.assertEqual(_ddl.parse_response_schema(schema), expected)

    def test_field_order_is_preserved(self):
        # JSON objects have no inherent order, but the properties are emitted
        # in declaration order so that the prompt and the schema read the same
        # way round.
        schema = _ddl.parse_response_schema("zebra STRING, apple INT")
        self.assertEqual(list(schema["properties"]), ["zebra", "apple"])


class TestOutput(unittest.TestCase):
    """Spot checks of the emitted schema, independent of Spark."""

    def test_scalar_fields(self):
        self.assertEqual(
            _ddl.parse_response_schema("a STRING, b INT"),
            {
                "type": "OBJECT",
                "properties": {
                    "a": {"type": "STRING"},
                    "b": {"type": "INTEGER"},
                },
            },
        )

    def test_not_null_becomes_required(self):
        self.assertEqual(
            _ddl.parse_response_schema("a STRING NOT NULL, b INT"),
            {
                "type": "OBJECT",
                "properties": {
                    "a": {"type": "STRING"},
                    "b": {"type": "INTEGER"},
                },
                "required": ["a"],
            },
        )

    def test_nullable_fields_are_not_required(self):
        self.assertNotIn(
            "required", _ddl.parse_response_schema("a STRING, b INT")
        )

    def test_array_of_struct(self):
        self.assertEqual(
            _ddl.parse_response_schema("xs ARRAY<STRUCT<a: STRING>>"),
            {
                "type": "OBJECT",
                "properties": {
                    "xs": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {"a": {"type": "STRING"}},
                        },
                    }
                },
            },
        )

    def test_backquoted_name_is_unquoted(self):
        schema = _ddl.parse_response_schema("`odd name` STRING")
        self.assertEqual(list(schema["properties"]), ["odd name"])


class TestRejects(unittest.TestCase):
    """Malformed or unsupported schemas must fail, and say why."""

    @classmethod
    def setUpClass(cls):
        # Needed only by the cross-check that Spark rejects the same names.
        cls.spark = SparkSession.builder.master("local[1]").getOrCreate()

    def test_rejects_malformed_schemas(self):
        cases = [
            "",
            "   ",
            "a",
            "STRING",
            "a STRING,",
            ", a STRING",
            "a STRUCT<>",
            "a ARRAY<>",
            "a STRUCT<b: STRING",
            "a ARRAY<STRING",
            "a STRING NOT",
            "a STRING COMMENT",
            "a STRING b STRING",
            "a STRING; DROP TABLE t",
        ]
        for schema in cases:
            with self.subTest(schema=schema):
                with self.assertRaises(ValueError):
                    _ddl.parse_response_schema(schema)

    def test_rejects_types_with_no_json_counterpart(self):
        # Rejected rather than approximated: silently turning a DECIMAL into a
        # float, or a TIMESTAMP into a string, would lose information the
        # caller asked for.
        for type_name in [
            "DECIMAL(10,2)",
            "NUMERIC",
            "BINARY",
            "DATE",
            "TIMESTAMP",
            "MAP<STRING, INT>",
            "VARCHAR(10)",
            "CHAR(3)",
        ]:
            with self.subTest(type_name=type_name):
                with self.assertRaises(ValueError) as context:
                    _ddl.parse_response_schema(f"a {type_name}")
                self.assertIn("output_schema", str(context.exception))

    def test_rejects_bigquery_type_names(self):
        # The contract is Spark's DDL. Spark's own parser rejects these too, so
        # accepting them here would be a silent divergence.
        for type_name in ["INT64", "FLOAT64", "BOOL", "BYTES"]:
            with self.subTest(type_name=type_name):
                with self.assertRaises(ValueError):
                    _ddl.parse_response_schema(f"a {type_name}")
                with self.assertRaises(Exception):
                    StructType.fromDDL(f"a {type_name}")

    def test_rejects_duplicate_field_names(self):
        with self.assertRaises(ValueError) as context:
            _ddl.parse_response_schema("a STRING, a INT")
        self.assertIn("duplicate", str(context.exception).lower())

    def test_rejects_duplicates_in_a_nested_struct(self):
        with self.assertRaises(ValueError):
            _ddl.parse_response_schema("s STRUCT<a: STRING, a: INT>")

    def test_rejects_schemas_nested_too_deeply(self):
        too_deep = "a " + "ARRAY<" * 6 + "STRING" + ">" * 6
        with self.assertRaises(ValueError) as context:
            _ddl.parse_response_schema(too_deep)
        self.assertIn("nested", str(context.exception).lower())

    def test_allows_schemas_at_the_nesting_limit(self):
        at_limit = "a " + "ARRAY<" * 3 + "STRING" + ">" * 3
        self.assertIsNotNone(_ddl.parse_response_schema(at_limit))

    def test_rejects_non_strings(self):
        for value in [None, 42, ["a STRING"], {"a": "STRING"}]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _ddl.parse_response_schema(value)

    def test_error_names_the_option_and_shows_the_input(self):
        with self.assertRaises(ValueError) as context:
            _ddl.parse_response_schema("a NOTATYPE")
        message = str(context.exception)
        self.assertIn("output_schema", message)
        self.assertIn("NOTATYPE", message)


if __name__ == "__main__":
    unittest.main()
