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

"""Multimodal prompt parts.

A prompt is either text or an ordered sequence of parts. In Spark the only
ordered container that can hold different types is a struct, so a multimodal
prompt is a struct whose fields are, in order, the parts of the prompt. A field
holding a string is text; a field holding a struct with a ``uri`` is a file.

By the time a struct column reaches a pandas UDF it has become a
``pandas.DataFrame`` with one column per field, and a nested struct field has
become a ``dict``. Parts are therefore read positionally and told apart by
their Python type. Field names are deliberately never consulted: an unnamed
part is called ``col1``, ``col2`` and so on, but a bare column keeps its own
name, so the names carry no reliable meaning.
"""

import mimetypes
import posixpath
from typing import Any, Dict, List, Optional, Sequence

#: The only field a file part must have.
URI_FIELD = "uri"
#: The optional field that overrides MIME detection.
CONTENT_TYPE_FIELD = "content_type"

_FILE_FIELD_SETS = (
    frozenset({URI_FIELD}),
    frozenset({URI_FIELD, CONTENT_TYPE_FIELD}),
)

# Types that Python's mimetypes module does not know about but that Gemini
# accepts. Checked only after the standard table misses.
_EXTRA_MIME_TYPES = {
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".avif": "image/avif",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mkv": "video/x-matroska",
}


def is_file_struct(field_names: Sequence[str]) -> bool:
    """Reports whether a struct is a lone file reference, not a part list.

    ``named_struct('uri', path)`` is the natural way to write a prompt that is
    a single file, but structurally it is indistinguishable from a part list
    holding one text part: both arrive as a one-column frame of strings. The
    shorthand is resolved in favour of the file reading, because a part list of
    exactly one string is what a plain string prompt is already for.

    Args:
        field_names: The struct's field names, in order.

    Returns:
        True if the struct should be read as one file reference.
    """
    return frozenset(field_names) in _FILE_FIELD_SETS


def to_file_part(value: Dict[str, Any]):
    """Converts one file struct into a Gemini part.

    Args:
        value: The struct's fields, as delivered by Arrow.

    Returns:
        A ``google.genai.types.Part`` referring to the file.

    Raises:
        ValueError: If the struct has no usable ``uri``, carries unknown
            fields, or names a file whose type cannot be determined.
    """
    unknown = set(value) - {URI_FIELD, CONTENT_TYPE_FIELD}
    if unknown:
        raise ValueError(
            "A file part accepts only 'uri' and 'content_type', got "
            f"{sorted(unknown)}."
        )

    uri = value.get(URI_FIELD)
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("A file part must have a non-empty 'uri' field.")
    uri = uri.strip()

    content_type = value.get(CONTENT_TYPE_FIELD)
    if _is_null(content_type):
        content_type = None
    elif not isinstance(content_type, str):
        raise ValueError(
            "A file part's 'content_type' must be a string, got "
            f"{type(content_type).__name__}."
        )
    mime_type = (content_type or "").strip() or guess_mime_type(uri)

    # Imported lazily so that importing this package does not pull in the
    # Vertex AI dependency until an AI function is evaluated.
    from google.genai import types  # pylint: disable=import-outside-toplevel

    return types.Part.from_uri(file_uri=uri, mime_type=mime_type)


def guess_mime_type(uri: str) -> str:
    """Infers a file's MIME type from its extension.

    Args:
        uri: The file's URI, for example ``gs://bucket/invoice.pdf``.

    Returns:
        The IANA MIME type.

    Raises:
        ValueError: If the extension is missing or unrecognized, in which case
            the caller has to say what the file is.
    """
    # Only the final path segment matters, and taking it avoids a query string
    # or a bucket name that happens to contain a dot being read as a suffix.
    name = posixpath.basename(uri.split("?", 1)[0].rstrip("/"))
    extension = posixpath.splitext(name)[1].lower()

    mime_type = _EXTRA_MIME_TYPES.get(extension)
    if mime_type:
        return mime_type

    mime_type, _ = mimetypes.guess_type(name)
    if mime_type:
        return mime_type

    raise ValueError(
        f"Could not determine the media type of '{uri}' from its name. Give "
        "it explicitly, for example "
        "named_struct('uri', path, 'content_type', 'application/pdf')."
    )


def to_contents(parts: List[Any]) -> Optional[List[Any]]:
    """Converts one row's parts into the ``contents`` of a request.

    Args:
        parts: The struct's field values for this row, in field order.

    Returns:
        The list of Gemini parts, or None if every part was null, which is
        treated like a null prompt.

    Raises:
        ValueError: If a part is neither text nor a file reference.
    """
    # Imported lazily for the same reason as in to_file_part.
    from google.genai import types  # pylint: disable=import-outside-toplevel

    contents = []
    for position, part in enumerate(parts):
        if _is_null(part):
            # A null part contributes nothing, exactly as an empty string
            # would. This keeps a row usable when only one of several optional
            # parts is missing.
            continue
        if isinstance(part, str):
            if part:
                contents.append(types.Part.from_text(text=part))
        elif isinstance(part, dict):
            if all(_is_null(field) for field in part.values()):
                # A file reference whose every field is null is a missing
                # part, not a malformed one: the row simply has no file.
                continue
            contents.append(to_file_part(part))
        else:
            raise ValueError(
                f"Prompt part {position + 1} must be text or a file struct "
                f"with a 'uri' field, got {type(part).__name__}."
            )

    return contents or None


def _is_null(value: Any) -> bool:
    """Reports whether an Arrow value is null.

    A struct field arrives as None, but a missing numeric arrives as NaN, and
    a dict is never null itself.
    """
    if value is None:
        return True
    return isinstance(value, float) and value != value
