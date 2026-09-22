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

"""A dependency-free parser for the ``output_schema`` DDL string.

``output_schema`` may be supplied as a runtime argument from Spark SQL, in
which case it reaches this library as a column value and has to be parsed on an
executor. Spark's own ``StructType.fromDDL`` cannot be used there: it calls
into the JVM through the active session, and a Python worker has none, so it
fails with ``SESSION_OR_CONTEXT_NOT_EXISTS``.

This module therefore parses the DDL itself, and goes straight to the Gemini
response schema without building a Spark type on the way, since the response
schema is all the caller needs. The grammar accepted here is deliberately the
subset that the rest of this package supports, and the unit tests assert that
it agrees with ``StructType.fromDDL`` on every case they cover, so the two
parsers cannot drift apart unnoticed.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# How deeply an output schema may nest. A deeply nested schema is far more
# often a mistake than an intention, and constrained decoding grows harder for
# the model with every level.
MAX_NESTING_DEPTH = 5

# The scalar types this package supports, in every spelling Spark's own DDL
# parser accepts, mapped to their Gemini response schema type. Spark has no
# single-precision or width-limited integer in JSON terms, so the whole integer
# family collapses to INTEGER and both floating point types to NUMBER.
#
# Types with no faithful JSON counterpart are deliberately absent: DECIMAL
# would silently lose precision, and BINARY, DATE, TIMESTAMP and INTERVAL have
# no schema-level representation the model can be constrained to. They are
# rejected rather than approximated.
_SCALAR_TYPES = {
    "STRING": "STRING",
    "BOOLEAN": "BOOLEAN",
    "BYTE": "INTEGER",
    "TINYINT": "INTEGER",
    "SHORT": "INTEGER",
    "SMALLINT": "INTEGER",
    "INT": "INTEGER",
    "INTEGER": "INTEGER",
    "LONG": "INTEGER",
    "BIGINT": "INTEGER",
    "FLOAT": "NUMBER",
    "REAL": "NUMBER",
    "DOUBLE": "NUMBER",
}

_SUPPORTED_TYPE_NAMES = (
    "STRING, BOOLEAN, TINYINT, SMALLINT, INT, BIGINT, FLOAT, DOUBLE, ARRAY "
    "and STRUCT"
)

_TOKEN_PATTERN = re.compile(
    r"""
      (?P<space>\s+)
    | (?P<quoted>`(?:[^`]|``)*`)
    | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
    | (?P<number>\d+)
    | (?P<punct>[<>,:()])
    | (?P<word>[A-Za-z_][A-Za-z_0-9]*)
    """,
    re.VERBOSE,
)


class _Token:
    """One lexical token, carrying its offset for error messages."""

    __slots__ = ("kind", "text", "position")

    def __init__(self, kind: str, text: str, position: int):
        self.kind = kind
        self.text = text
        self.position = position

    @property
    def upper(self) -> str:
        return self.text.upper()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Token({self.kind!r}, {self.text!r}, {self.position})"


def parse_response_schema(output_schema: str) -> Dict[str, Any]:
    """Parses an ``output_schema`` string into a Gemini response schema.

    Args:
        output_schema: A comma separated list of fields, such as
            ``"sentiment STRING, score INT64"``. A schema wrapped in
            ``STRUCT<...>`` is also accepted.

    Returns:
        A schema dictionary suitable for the ``response_schema`` generation
        setting, with the fields in the order they were declared.

    Raises:
        ValueError: If the schema cannot be parsed, uses an unsupported type,
            declares duplicate fields, or nests too deeply.
    """
    if not isinstance(output_schema, str):
        raise ValueError(
            "output_schema must be a string, got "
            f"{type(output_schema).__name__}."
        )
    return _Parser(output_schema).parse()


def _tokenize(text: str) -> List[_Token]:
    """Splits the schema into tokens, rejecting anything unrecognized."""
    tokens: List[_Token] = []
    position = 0
    while position < len(text):
        match = _TOKEN_PATTERN.match(text, position)
        if match is None:
            raise _error(
                text, position, f"unexpected character {text[position]!r}"
            )
        position = match.end()
        kind = match.lastgroup
        if kind == "space":
            continue
        tokens.append(_Token(kind, match.group(), match.start()))
    return tokens


def _error(source: str, position: int, detail: str) -> ValueError:
    """Builds the error raised for any malformed schema."""
    return ValueError(
        "Failed to parse the option `output_schema` at position "
        f"{position}: {detail}. It must be a valid comma separated list of "
        "field names and types, for example 'sentiment STRING, score INT64'. "
        f"Got: {source!r}."
    )


class _Parser:
    """A recursive descent parser over the supported DDL subset."""

    def __init__(self, source: str):
        self._source = source
        self._tokens = _tokenize(source)
        self._index = 0

    # -- token helpers ----------------------------------------------------

    def _peek(self) -> Optional[_Token]:
        if self._index < len(self._tokens):
            return self._tokens[self._index]
        return None

    def _next(self) -> _Token:
        token = self._peek()
        if token is None:
            raise self._fail("the schema ended unexpectedly")
        self._index += 1
        return token

    def _at_punct(self, text: str) -> bool:
        token = self._peek()
        return (
            token is not None
            and token.kind == "punct"
            and token.text == text
        )

    def _at_word(self, word: str) -> bool:
        token = self._peek()
        return (
            token is not None
            and token.kind == "word"
            and token.upper == word
        )

    def _take_punct(self, text: str) -> None:
        if not self._at_punct(text):
            raise self._fail(f"expected {text!r}")
        self._index += 1

    def _fail(self, detail: str) -> ValueError:
        token = self._peek()
        position = (
            token.position if token is not None else len(self._source)
        )
        found = f", found {token.text!r}" if token is not None else ""
        return _error(self._source, position, f"{detail}{found}")

    # -- grammar ----------------------------------------------------------

    def parse(self) -> Dict[str, Any]:
        """Parses the whole schema and checks that nothing is left over."""
        # A schema may be written either as a bare field list or wrapped in
        # STRUCT<...>, both of which Spark's own parser accepts.
        if self._at_word("STRUCT"):
            schema = self._parse_type(depth=1)
            if self._peek() is not None:
                raise self._fail("expected the end of the schema")
            if schema.get("type") != "OBJECT":
                raise self._fail("expected a struct")
            return schema

        fields = self._parse_field_list(depth=1, terminator=None)
        if self._peek() is not None:
            raise self._fail("expected the end of the schema")
        return _to_object(fields)

    def _parse_field_list(
        self, depth: int, terminator: Optional[str]
    ) -> List[Tuple[str, Dict[str, Any], bool]]:
        """Parses one or more comma separated fields.

        Returns:
            A list of ``(name, schema, required)`` triples in declaration
            order.
        """
        fields = []
        while True:
            fields.append(self._parse_field(depth))
            if self._at_punct(","):
                self._index += 1
                continue
            break

        if terminator is not None:
            self._take_punct(terminator)

        names = [name for name, _, _ in fields]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            listed = ", ".join(duplicates)
            raise ValueError(
                f"Option `output_schema` has duplicate field names: {listed}."
            )
        if not fields:
            raise ValueError("output_schema must declare at least one field.")
        return fields

    def _parse_field(
        self, depth: int
    ) -> Tuple[str, Dict[str, Any], bool]:
        """Parses ``name [:] type [NOT NULL] [COMMENT '...']``."""
        name = self._parse_identifier()
        # Inside STRUCT<> Spark allows, but does not require, a colon between
        # the field name and its type.
        if self._at_punct(":"):
            self._index += 1
        schema = self._parse_type(depth)

        required = False
        if self._at_word("NOT"):
            self._index += 1
            if not self._at_word("NULL"):
                raise self._fail("expected NULL after NOT")
            self._index += 1
            required = True

        if self._at_word("COMMENT"):
            self._index += 1
            token = self._peek()
            if token is None or token.kind != "string":
                raise self._fail("expected a quoted string after COMMENT")
            self._index += 1

        return name, schema, required

    def _parse_identifier(self) -> str:
        """Parses a bare or backquoted field name."""
        token = self._next()
        if token.kind == "word":
            return token.text
        if token.kind == "quoted":
            # Spark escapes a backquote inside a quoted identifier by doubling
            # it.
            return token.text[1:-1].replace("``", "`")
        raise _error(
            self._source,
            token.position,
            f"expected a field name, found {token.text!r}",
        )

    def _parse_type(self, depth: int) -> Dict[str, Any]:
        """Parses a scalar, ARRAY or STRUCT type."""
        if depth > MAX_NESTING_DEPTH:
            raise ValueError(
                "Option `output_schema` has too many nested levels. Maximum "
                f"allowed is {MAX_NESTING_DEPTH}."
            )

        token = self._next()
        if token.kind != "word":
            raise _error(
                self._source,
                token.position,
                f"expected a type name, found {token.text!r}",
            )

        name = token.upper
        if name == "ARRAY":
            self._take_punct("<")
            element = self._parse_type(depth + 1)
            self._take_punct(">")
            return {"type": "ARRAY", "items": element}

        if name == "STRUCT":
            self._take_punct("<")
            fields = self._parse_field_list(depth + 1, terminator=">")
            return _to_object(fields)

        if name in _SCALAR_TYPES:
            return {"type": _SCALAR_TYPES[name]}

        raise _error(
            self._source,
            token.position,
            f"unsupported field type {token.text!r}. Only "
            f"{_SUPPORTED_TYPE_NAMES} are supported",
        )


def _to_object(
    fields: List[Tuple[str, Dict[str, Any], bool]]
) -> Dict[str, Any]:
    """Builds an OBJECT response schema from parsed fields."""
    schema: Dict[str, Any] = {
        "type": "OBJECT",
        "properties": {name: field for name, field, _ in fields},
    }
    required = [name for name, _, is_required in fields if is_required]
    if required:
        schema["required"] = required
    return schema
