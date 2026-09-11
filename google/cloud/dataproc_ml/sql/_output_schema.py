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

"""Translation of the ``output_schema`` option into Spark and Gemini schemas.

A single schema declaration drives two things: the Spark ``StructType`` that
the AI function returns, and the response schema sent to the model so that it
replies with conforming JSON.
"""

import re
from typing import Any, Dict, Union

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# BigQuery spells some scalar types differently from Spark. Both spellings are
# accepted so that a schema can be copied straight from a BigQuery query.
_TYPE_ALIASES = {
    "INT64": "BIGINT",
    "FLOAT64": "DOUBLE",
    "BOOL": "BOOLEAN",
}

# An alias is only rewritten where a type may appear, which is recognized by
# what surrounds it. Matching the bare name would also rewrite a *field* named
# after a type: "int64 INT64" would become "BIGINT BIGINT", quietly renaming
# the field to BIGINT instead of failing. A type is preceded by a space, a
# colon or "<", and is followed by the end of the field or by one of the
# modifiers that Spark allows after a type.
_ALIAS_PATTERNS = tuple(
    (
        re.compile(
            rf"(?<=[\s:<]){bigquery_name}"
            r"(?=\s*(?:,|>|\)|$|NOT\s+NULL|COMMENT\b))",
            re.IGNORECASE,
        ),
        spark_name,
    )
    for bigquery_name, spark_name in _TYPE_ALIASES.items()
)

# Matches the field options syntax that BigQuery supports but this library does
# not parse yet, e.g. ``OPTIONS(description = '...')``.
_OPTIONS_PATTERN = re.compile(r"\bOPTIONS\s*\(", re.IGNORECASE)

# Mirrors BigQuery's limit on how deeply an output schema may nest.
MAX_NESTING_DEPTH = 5

_SCALAR_RESPONSE_TYPES = {
    StringType: "STRING",
    BooleanType: "BOOLEAN",
    LongType: "INTEGER",
    IntegerType: "INTEGER",
    DoubleType: "NUMBER",
    FloatType: "NUMBER",
}

_SUPPORTED_TYPE_NAMES = "STRING, INT64, FLOAT64, BOOL, ARRAY and STRUCT"


def parse_output_schema(
    output_schema: Union[str, StructType],
) -> StructType:
    """Parses the ``output_schema`` option into a Spark ``StructType``.

    Args:
        output_schema: Either a comma separated list of fields, such as
            ``"sentiment STRING, score INT64"``, or an already built
            ``StructType``.

    Returns:
        The validated struct, with fields sorted by name.

    Raises:
        ValueError: If the schema cannot be parsed, uses an unsupported type,
            or nests too deeply.
    """
    if isinstance(output_schema, StructType):
        struct = output_schema
    elif isinstance(output_schema, str):
        struct = _parse_ddl(output_schema)
    else:
        raise ValueError(
            "output_schema must be a string or a pyspark StructType, got "
            f"{type(output_schema).__name__}."
        )

    if not struct.fields:
        raise ValueError("output_schema must declare at least one field.")

    _validate(struct, depth=1)
    return _sorted_struct(struct)


def _parse_ddl(output_schema: str) -> StructType:
    """Parses the schema string, translating BigQuery type names first."""
    if _OPTIONS_PATTERN.search(output_schema):
        raise ValueError(
            "The OPTIONS clause in output_schema is not supported yet. "
            "Remove it and describe the fields in the prompt instead."
        )

    normalized = output_schema
    for pattern, spark_name in _ALIAS_PATTERNS:
        normalized = pattern.sub(spark_name, normalized)

    try:
        parsed = StructType.fromDDL(normalized)
    except Exception as e:
        raise ValueError(
            "Failed to parse the option `output_schema`. It must be a valid "
            "comma separated list of field names and types, for example "
            f"'sentiment STRING, score INT64'. Got: {output_schema!r}."
        ) from e

    if not isinstance(parsed, StructType):
        raise ValueError(
            "Failed to parse the option `output_schema` into a struct: "
            f"{output_schema!r}."
        )
    return parsed


def _validate(data_type: DataType, depth: int) -> None:
    """Recursively checks that only supported types are used."""
    if depth > MAX_NESTING_DEPTH:
        raise ValueError(
            "Option `output_schema` has too many nested levels. Maximum "
            f"allowed is {MAX_NESTING_DEPTH}."
        )

    if isinstance(data_type, StructType):
        names = [field.name for field in data_type.fields]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(
                f"Option `output_schema` has duplicate field names: {names}."
            )
        for field in data_type.fields:
            _validate(field.dataType, depth + 1)
    elif isinstance(data_type, ArrayType):
        _validate(data_type.elementType, depth + 1)
    elif type(data_type) not in _SCALAR_RESPONSE_TYPES:
        raise ValueError(
            "Unsupported field type in output_schema: "
            f"{data_type.simpleString()}."
            f" Only {_SUPPORTED_TYPE_NAMES} are supported."
        )


def _sorted_struct(struct: StructType) -> StructType:
    """Returns the struct with its fields, and nested fields, sorted by name.

    BigQuery returns the fields of a custom output schema in alphabetical
    order rather than declaration order, and this library matches it so that
    the same query produces the same column layout on both engines.
    """
    return StructType(
        [
            StructField(
                field.name,
                _sorted_type(field.dataType),
                nullable=field.nullable,
            )
            for field in sorted(struct.fields, key=lambda f: f.name)
        ]
    )


def _sorted_type(data_type: DataType) -> DataType:
    """Applies :func:`_sorted_struct` to any nested structs."""
    if isinstance(data_type, StructType):
        return _sorted_struct(data_type)
    if isinstance(data_type, ArrayType):
        return ArrayType(
            _sorted_type(data_type.elementType),
            containsNull=data_type.containsNull,
        )
    return data_type


def to_response_schema(struct: StructType) -> Dict[str, Any]:
    """Converts a Spark struct into a Gemini response schema.

    Args:
        struct: The parsed output schema.

    Returns:
        A schema dictionary suitable for the ``response_schema`` generation
        setting.
    """
    return _to_response_schema(struct)


def _to_response_schema(data_type: DataType) -> Dict[str, Any]:
    """Recursively converts a Spark type into a Gemini response schema."""
    if isinstance(data_type, StructType):
        required = [
            field.name for field in data_type.fields if not field.nullable
        ]
        schema: Dict[str, Any] = {
            "type": "OBJECT",
            "properties": {
                field.name: _to_response_schema(field.dataType)
                for field in data_type.fields
            },
        }
        if required:
            schema["required"] = required
        return schema

    if isinstance(data_type, ArrayType):
        return {
            "type": "ARRAY",
            "items": _to_response_schema(data_type.elementType),
        }

    return {"type": _SCALAR_RESPONSE_TYPES[type(data_type)]}
