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

"""``AI.GENERATE``: generative text and structured output over Spark columns."""

import asyncio
import dataclasses
import functools
import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from pyspark.sql import Column
from pyspark.sql import functions as sf
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import StringType, StructField, StructType

from google.cloud.dataproc_ml.sql import _client
from google.cloud.dataproc_ml.sql import _ddl
from google.cloud.dataproc_ml.sql import _endpoint
from google.cloud.dataproc_ml.sql import _model_params
from google.cloud.dataproc_ml.sql import _parts
from google.cloud.dataproc_ml.sql import _retry

logger = logging.getLogger(__name__)

#: The model used when no endpoint is given.
DEFAULT_ENDPOINT = "gemini-3.6-flash"

# Fixed request settings. These will be configurable in a follow-up PR.
_MAX_CONCURRENT_REQUESTS = 5
_MAX_ATTEMPTS = 5
_BASE_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 60.0

# How many distinct configurations to keep resolved. A query uses one per
# distinct set of settings, and the settings are constant for a query.
_CONFIG_CACHE_SIZE = 64

RESULT_FIELD = "result"
FULL_RESPONSE_FIELD = "full_response"
STATUS_FIELD = "status"

#: ``status`` when the row succeeded. An empty string rather than a word, so
#: that the dead-letter query is ``WHERE status <> ''`` and a caller never has
#: to know the vocabulary of success. A null prompt also reports success, with
#: a null result.
STATUS_SUCCESS = ""

#: ``full_response`` when there is no model response to report. The field is
#: never null, so ``parse_json(full_response)`` is safe on every row.
EMPTY_RESPONSE = "{}"

#: Leads the ``status`` of a row the model refused to answer.
STATUS_SAFETY_BLOCKED = "SAFETY_BLOCKED"

#: Leads the ``status`` of a row whose prompt could not be understood.
STATUS_INVALID_ARGUMENT = "INVALID_ARGUMENT"

#: The struct every call returns, whatever the arguments. Keeping the type
#: fixed is what lets ``output_schema`` be an ordinary runtime argument: the
#: query can be planned without knowing its value.
RETURN_TYPE = StructType(
    [
        StructField(RESULT_FIELD, StringType()),
        StructField(FULL_RESPONSE_FIELD, StringType()),
        StructField(STATUS_FIELD, StringType()),
    ]
)

_RETURN_FIELDS = [field.name for field in RETURN_TYPE.fields]

# Finish reasons that mean the content was withheld rather than generated.
_BLOCKED_FINISH_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
)

# The settings that describe the call rather than the row. Spark SQL cannot
# tell a literal from a column at plan time, so constancy is checked here.
_CONSTANT_ARGUMENTS = ("endpoint", "model_params", "output_schema")


@dataclasses.dataclass(frozen=True)
class FilePart:
    """A file to include in a prompt. Build one with :func:`file`."""

    uri: Column
    content_type: Optional[Column] = None


PromptPart = Union[Column, str, FilePart]
PromptArg = Union[PromptPart, Sequence[PromptPart]]
ModelParamsArg = Union[str, Dict[str, Any]]
OutputSchemaArg = Union[str, StructType]


def file(uri: Union[Column, str], content_type=None) -> FilePart:
    """References a file to be sent to the model as part of a prompt.

    The file is read by Vertex AI, not by Spark, so the service account
    running the job needs read access to it.

    Following the convention of the rest of this API, a plain string names a
    *column*; wrap a literal path in :func:`pyspark.sql.functions.lit`.

    Args:
        uri: The file's URI, such as ``gs://bucket/invoice.pdf``, as a column
            or the name of one.
        content_type: The file's media type, for example
            ``application/pdf``, as a column, the name of one, or a plain
            string. Inferred from the file extension when not given.

    Returns:
        A prompt part that :func:`ai_generate` turns into a file reference.

    Example:
        >>> df.withColumn(
        ...     "g",
        ...     ai_generate(
        ...         [sf.lit("What is the invoice total?"), file(sf.col("uri"))]
        ...     ),
        ... )
    """
    return FilePart(uri=uri, content_type=content_type)


def ai_generate(
    prompt: PromptArg,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model_params: Optional[ModelParamsArg] = None,
    output_schema: Optional[OutputSchemaArg] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    _generate_content=None,
) -> Column:
    """Generates text for every row using a Gemini model.

    The prompt follows the usual PySpark convention: a string names a column,
    and literal text must be wrapped in :func:`pyspark.sql.functions.lit`. A
    sequence of parts forms one prompt; text parts are concatenated in order,
    and :func:`file` includes a file.

    Requests are retried with exponential backoff and are issued a few at a
    time per worker.

    .. note::
        Using this function incurs Vertex AI charges. See the
        `Vertex AI Generative AI pricing
        <https://cloud.google.com/vertex-ai/generative-ai/pricing>`_ page.

    .. note::
        Requires Apache Spark 4.0 or later.

    Args:
        prompt: The prompt column, the name of a prompt column, a
            :func:`file`, or a sequence of those forming one prompt.
        endpoint: The Gemini model to call, either a name such as
            ``gemini-3.6-flash`` or a fully qualified
            ``projects/.../publishers/google/models/...`` path. Defaults to
            ``gemini-3.6-flash``. The name is checked against the model
            catalogue before any generation, so a typo fails the query rather
            than filling a column with errors.
        model_params: Additional request settings, as a JSON string or an
            equivalent dictionary, following the Vertex AI ``generateContent``
            request body without its ``contents`` field, for example
            ``'{"generation_config": {"temperature": 0.2}}'``.
        output_schema: The shape the model must answer in, as a Spark DDL
            string such as ``"sentiment STRING, score INT"`` or as a
            ``StructType``. The model is constrained to it, and ``result``
            holds the matching JSON document rather than free text.
        project: The Google Cloud project billed for the calls. Inferred from
            the environment when not set.
        location: The Vertex AI location. Defaults to ``global``, which is
            where the current Gemini models are served. Set it only when a
            particular region is required, and check that the endpoint is
            available there.

    Returns:
        A column holding a ``STRUCT<result STRING, full_response STRING,
        status STRING>``. ``result`` is the generated text, or a JSON document
        when ``output_schema`` is set, and is null when the row failed.
        ``full_response`` is the whole model response as JSON, or ``"{}"``
        when there is none. ``status`` is empty on success and otherwise
        describes the failure, led by its canonical error code, so that
        ``WHERE g.status <> ''`` collects the rows that need attention.

    Raises:
        ValueError: If any argument is invalid. The arguments given here are
            validated on the driver, so mistakes surface before a stage is
            scheduled.

    Example:
        >>> df.withColumn("g", ai_generate(sf.col("review"))).select(
        ...     "g.result", "g.status"
        ... )
        >>> df.withColumn(
        ...     "g",
        ...     ai_generate(
        ...         [sf.lit("Summarize in 10 words: "), sf.col("body")],
        ...         output_schema="summary STRING, sentiment STRING",
        ...     ),
        ... )
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty model name.")

    # Resolving the configuration here, rather than only on the executor,
    # turns a malformed model_params or output_schema into an immediate error
    # instead of a failed stage.
    model_params_json = _to_model_params_json(model_params)
    output_schema_ddl = _to_output_schema_ddl(output_schema)
    _resolve_config(model_params_json, output_schema_ddl)

    generate = ai_generate_udf(
        project=project,
        location=location,
        _generate_content=_generate_content,
    )
    # An unset argument is passed as a typed NULL rather than left out, so
    # that the executor always sees the same four columns. Without the cast an
    # unset one would be a void column, which is not a string.
    return generate(
        _to_prompt_column(prompt),
        sf.lit(endpoint).cast(StringType()),
        sf.lit(model_params_json).cast(StringType()),
        sf.lit(output_schema_ddl).cast(StringType()),
    )


def ai_generate_udf(
    *,
    project: Optional[str] = None,
    location: Optional[str] = None,
    _generate_content=None,
):
    """Builds the ``AI.GENERATE`` user defined function for Spark SQL.

        >>> spark.udf.register("ai_generate", ai_generate_udf())
        >>> spark.sql('''
        ...     SELECT ai_generate(
        ...         prompt => concat('Summarize: ', body),
        ...         endpoint => 'gemini-3.6-flash',
        ...         output_schema => 'summary STRING, sentiment STRING'
        ...     ).result
        ...     FROM articles
        ... ''')

    Unlike :func:`ai_generate`, the model settings are not fixed when the
    function is built. They are ordinary arguments, so they may be omitted or
    passed by name in any order — which requires Apache Spark 4.0 or later.
    They must still be constant for the query: ``endpoint``, ``model_params``
    and ``output_schema`` describe the call, not the row, and a value that
    varies from row to row fails the query.

    ``prompt`` is the one argument that varies. It is either a string or a
    struct whose fields are, in order, the parts of the prompt: a string field
    is text, and a field holding ``named_struct('uri', ...)`` is a file. A
    top-level ``named_struct('uri', ...)`` is itself a single file.

    Args:
        project: The Google Cloud project billed for the calls. Inferred from
            the environment when not set.
        location: The Vertex AI location. See :func:`ai_generate`.
        _generate_content: Internal. Replaces the call to Vertex AI with an
            awaitable of the same shape, so that tests can run the whole Spark
            path without credentials or network access.

    Returns:
        A pandas user defined function taking ``prompt`` and, optionally,
        ``endpoint``, ``model_params`` and ``output_schema``.
    """

    @pandas_udf(RETURN_TYPE)
    def _generate(
        prompt: Union[pd.Series, pd.DataFrame],
        endpoint: pd.Series = None,
        model_params: pd.Series = None,
        output_schema: pd.Series = None,
    ) -> pd.DataFrame:
        """Runs on the executor, once per Arrow batch.

        Spark omits an argument the caller did not supply, which arrives here
        as ``None`` rather than as a column of nulls. That is the only way to
        tell "not given" from "given as NULL", so it is what selects the
        defaults.

        A struct prompt arrives as a ``pandas.DataFrame`` with one column per
        field; a string prompt arrives as a ``pandas.Series``.
        """
        model = _constant(endpoint, "endpoint") or DEFAULT_ENDPOINT
        # A malformed setting is a mistake in the query, not a property of a
        # row, so it is raised rather than written to every row's status.
        config = _resolve_config(
            _constant(model_params, "model_params"),
            _constant(output_schema, "output_schema"),
        )

        if _generate_content is not None:
            generate_content = _generate_content
        else:
            # Cached for the life of the worker process, so this is a lookup
            # rather than a new connection pool per batch.
            client = _client.get_client(project, location)
            _endpoint.check(client, model, project, location)
            generate_content = client.aio.models.generate_content

        rows = _client.run_coroutine(
            _generate_batch(
                generate_content=generate_content,
                prompts=_to_prompts(prompt),
                model=model,
                config=config,
            )
        )
        return pd.DataFrame(rows, columns=_RETURN_FIELDS, index=prompt.index)

    return _generate


def _constant(argument: Optional[pd.Series], name: str) -> Optional[str]:
    """Returns the single value of a setting that must not vary by row.

    ``endpoint``, ``model_params`` and ``output_schema`` configure the call,
    and BigQuery's ``AI.GENERATE`` likewise accepts only a literal for each.
    Spark SQL cannot tell a literal from a column that happens to repeat, so
    the rule is enforced here, per batch, where the values are visible.

    Args:
        argument: The argument's column, or None if it was not supplied.
        name: The argument's name, for the error message.

    Returns:
        The value, or None if it was absent, null or blank.

    Raises:
        ValueError: If the batch contains more than one distinct value.
    """
    if argument is None or argument.empty:
        return None
    if argument.nunique(dropna=False) > 1:
        seen = sorted(str(value) for value in argument.unique()[:3])
        raise ValueError(
            f"{name} must be the same for every row: it configures the call, "
            f"not the row, so it has to be a literal. Saw {seen}."
        )
    return _argument_text(argument.iloc[0])


def _to_prompts(prompt: Union[pd.Series, "pd.DataFrame"]) -> List[Any]:
    """Turns one batch's prompt column into one prompt value per row.

    Args:
        prompt: The prompt column, as a Series of strings or as a DataFrame
            whose columns are the fields of a struct.

    Returns:
        One entry per row: a string, or a list of parts for a struct prompt.
    """
    if not isinstance(prompt, pd.DataFrame):
        return prompt.tolist()

    if _parts.is_file_struct(list(prompt.columns)):
        # ``named_struct('uri', path)`` is a single file, not a part list
        # holding one string. See _parts.is_file_struct.
        return [[fields] for fields in prompt.to_dict(orient="records")]

    # Field order is column order; the names are not meaningful. ``name=None``
    # yields plain tuples, which avoids the cost and the name mangling of
    # building a namedtuple class per batch.
    return [list(row) for row in prompt.itertuples(index=False, name=None)]


def _to_model_params_json(
    model_params: Optional[ModelParamsArg],
) -> Optional[str]:
    """Renders the ``model_params`` argument as the JSON string the UDF takes.

    The user facing function accepts a dictionary as well, because building
    one in Python is more natural than quoting JSON, but only one form has to
    survive the trip to an executor.
    """
    if model_params is None:
        return None
    if isinstance(model_params, str):
        return model_params
    if isinstance(model_params, dict):
        try:
            return json.dumps(model_params)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "model_params must be JSON serializable. Cause: " f"{e}"
            ) from e
    raise ValueError(
        "model_params must be a JSON string or a dictionary, got "
        f"{type(model_params).__name__}."
    )


def _to_output_schema_ddl(
    output_schema: Optional[OutputSchemaArg],
) -> Optional[str]:
    """Renders the ``output_schema`` argument as a DDL string.

    A ``StructType`` is accepted for convenience in Python and rendered with
    Spark's own ``toDDL``, so that the executor only ever sees DDL and only
    one parser has to agree with Spark.
    """
    if output_schema is None:
        return None
    if isinstance(output_schema, str):
        return output_schema
    if isinstance(output_schema, StructType):
        return output_schema.toDDL()
    raise ValueError(
        "output_schema must be a DDL string or a pyspark StructType, got "
        f"{type(output_schema).__name__}."
    )


@functools.lru_cache(maxsize=_CONFIG_CACHE_SIZE)
def _resolve_config(
    model_params_json: Optional[str], output_schema_ddl: Optional[str]
):
    """Builds the generation config for one combination of arguments.

    The cache turns the per-batch parse into a single parse per distinct
    configuration for the life of the worker. The result is only read, never
    mutated, so sharing one config across rows and threads is safe.

    Args:
        model_params_json: The ``model_params`` argument, or None.
        output_schema_ddl: The ``output_schema`` argument, or None.

    Returns:
        A ``google.genai.types.GenerateContentConfig``.

    Raises:
        ValueError: If either argument is invalid.
    """
    response_schema = (
        _ddl.parse_response_schema(output_schema_ddl)
        if output_schema_ddl
        else None
    )
    flat_params = _model_params.normalize_model_params(
        model_params_json, response_schema
    )
    return _model_params.build_generation_config(flat_params)


def _to_prompt_column(prompt: PromptArg) -> Column:
    """Builds the single column that holds each row's prompt.

    A text-only prompt becomes one STRING column, concatenated if it was given
    in parts. A prompt that includes a :func:`file` becomes a struct whose
    fields are the parts in order, which is the only ordered container SQL has
    that can hold values of different types.

    Every text path casts to STRING. A prompt column of another type would
    otherwise reach the worker as a non-string value and be treated as a null
    prompt, turning a whole column into nulls without any error.
    """
    if isinstance(prompt, FilePart):
        return _to_file_column(prompt)
    if isinstance(prompt, (Column, str)):
        return _to_text_column(prompt)
    if isinstance(prompt, Sequence):
        parts = list(prompt)
        if not parts:
            raise ValueError("prompt must contain at least one part.")
        if len(parts) == 1:
            return _to_prompt_column(parts[0])
        if not any(isinstance(part, FilePart) for part in parts):
            return sf.concat(*[_to_text_column(part) for part in parts])
        # Aliasing the fields keeps them from accidentally spelling a file
        # reference, and keeps the struct's own names out of the way: the
        # worker reads parts by position, never by name.
        return sf.struct(
            *[
                _to_part_column(part).alias(f"_{position}")
                for position, part in enumerate(parts, start=1)
            ]
        )
    raise ValueError(
        "prompt must be a Column, a column name, a file(), or a sequence of "
        f"those, got {type(prompt).__name__}."
    )


def _to_part_column(part: PromptPart) -> Column:
    """Builds the column for one part of a multimodal prompt."""
    if isinstance(part, FilePart):
        return _to_file_column(part)
    return _to_text_column(part)


def _to_text_column(part: Union[Column, str]) -> Column:
    """Builds the STRING column for one text part."""
    if isinstance(part, Column):
        return part.cast(StringType())
    if isinstance(part, str):
        return sf.col(part).cast(StringType())
    raise ValueError(
        "A text prompt part must be a Column or a column name, got "
        f"{type(part).__name__}."
    )


def _to_file_column(part: FilePart) -> Column:
    """Builds the struct column that stands for one file reference.

    The field names are the ones the worker looks for, and are what makes a
    top-level file prompt recognizable.
    """
    fields = [_to_text_column(part.uri).alias(_parts.URI_FIELD)]
    if part.content_type is not None:
        content_type = part.content_type
        if isinstance(content_type, str):
            # Unlike a prompt, a media type is a constant far more often than
            # it is a column, and 'application/pdf' cannot name a column.
            content_type = sf.lit(content_type)
        fields.append(
            content_type.cast(StringType()).alias(_parts.CONTENT_TYPE_FIELD)
        )
    return sf.struct(*fields)


async def _generate_batch(
    *,
    generate_content,
    prompts: List[Any],
    model: str,
    config,
) -> List[Dict[str, Any]]:
    """Generates a response for every prompt in an Arrow batch."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

    async def _one(prompt: Any) -> Dict[str, Any]:
        async with semaphore:
            return await _generate_one(
                generate_content=generate_content,
                prompt=prompt,
                model=model,
                config=config,
            )

    return await asyncio.gather(*[_one(prompt) for prompt in prompts])


async def _generate_one(
    *,
    generate_content,
    prompt: Any,
    model: str,
    config,
) -> Dict[str, Any]:
    """Generates the response for one row, retrying transient failures."""
    try:
        contents = _to_contents(prompt)
    except ValueError as e:
        # A malformed part is a property of the row, unlike a malformed
        # setting, so it is reported rather than raised.
        return _row(result=None, status=f"{STATUS_INVALID_ARGUMENT}: {e}")

    if contents is None:
        # A null prompt produces a null result without calling the model.
        return _row(result=None, status=STATUS_SUCCESS)

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await generate_content(
                model=model, contents=contents, config=config
            )
            return _to_row(response)
        except Exception as e:  # pylint: disable=broad-except
            # Anything that is not a backend failure, such as a bug in this
            # library, is left to fail the task instead of being hidden in the
            # status field.
            if not _retry.is_reportable(e):
                raise
            is_last_attempt = attempt == _MAX_ATTEMPTS - 1
            if _retry.is_retryable(e) and not is_last_attempt:
                delay = _retry.full_jitter_delay(
                    attempt,
                    _BASE_RETRY_DELAY_SECONDS,
                    _MAX_RETRY_DELAY_SECONDS,
                )
                logger.warning(
                    "Retrying request to %s in %.2fs after error: %s",
                    model,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue
            return _row(result=None, status=_retry.to_status(e))

    # Unreachable: the loop either returns a row or retries.
    raise AssertionError("The retry loop ended without producing a row.")


def _to_contents(prompt: Any) -> Optional[Any]:
    """Renders one row's prompt as the ``contents`` of a request.

    Args:
        prompt: A string, a list of parts, or a null.

    Returns:
        The request contents, or None if the row has no prompt at all.

    Raises:
        ValueError: If a part is neither text nor a file reference.
    """
    if isinstance(prompt, list):
        return _parts.to_contents(prompt)
    if _is_missing(prompt):
        return None
    # ``ai_generate`` casts its prompt column to STRING on the driver, but the
    # function returned by ``ai_generate_udf`` can also be registered and
    # called straight from Spark SQL, where no such cast is inserted. Render
    # the value here so that a non-string column, say an id, still produces a
    # prompt rather than being dropped.
    return _to_text(prompt)


def _is_missing(value: Any) -> bool:
    """Reports whether a value is null.

    Spark nulls arrive as ``None``, as NaN in a numeric batch, or as
    ``pandas.NA`` from a nullable column. Every other value, including a
    number or a timestamp, is a real value.
    """
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)


def _argument_text(value: Any) -> Optional[str]:
    """Normalizes the value of a string setting.

    A setting that was not supplied, was supplied as NULL, or was supplied as
    an empty string all mean the same thing: use the default.
    """
    if _is_missing(value):
        return None
    text = _to_text(value).strip()
    return text or None


def _to_text(value: Any) -> str:
    """Renders a value that Spark has not already cast to a string.

    A whole number is rendered without a fraction. Arrow hands a nullable
    integer column to pandas as floats, so a BIGINT 2 arrives here as 2.0 as
    soon as any row in the batch is null. Rendering that as "2.0" would let
    one row's prompt depend on whether an unrelated row happened to be null.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _to_row(response) -> Dict[str, Any]:
    """Converts a model response into one output row."""
    full_response = _dump_response(response)

    blocked = _blocked_reason(response)
    if blocked:
        return _row(
            result=None,
            full_response=full_response,
            status=f"{STATUS_SAFETY_BLOCKED}: the model withheld the "
            f"response ({blocked}).",
        )

    # When output_schema is set the model is constrained to it, so the text is
    # already a JSON document conforming to the schema and is passed through
    # unchanged rather than being parsed and rebuilt.
    return _row(
        result=_extract_text(response),
        full_response=full_response,
        status=STATUS_SUCCESS,
    )


def _row(
    *,
    result: Optional[str],
    status: str,
    full_response: str = EMPTY_RESPONSE,
) -> Dict[str, Any]:
    """Builds an output row.

    ``full_response`` defaults to an empty JSON document rather than to null,
    so that every row can be fed to ``parse_json`` and so that a query cannot
    tell a failure apart from a missing column.
    """
    return {
        RESULT_FIELD: result,
        FULL_RESPONSE_FIELD: full_response,
        STATUS_FIELD: status,
    }


def _blocked_reason(response) -> Optional[str]:
    """Returns why the model withheld the answer, or None if it did not.

    A block is reported either on the prompt, before anything is generated, or
    as the finish reason of the candidate.
    """
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        return _reason_name(block_reason)

    for candidate in getattr(response, "candidates", None) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is None:
            continue
        name = _reason_name(reason)
        if name.upper() in _BLOCKED_FINISH_REASONS:
            return name
    return None


def _reason_name(reason) -> str:
    """Names a reason the client library reports as an enum.

    A plain string is accepted too, so that a test or a future version of the
    library needs no special handling.
    """
    return getattr(reason, "name", None) or str(reason)


def _extract_text(response) -> Optional[str]:
    """Returns the generated text.

    Only the first candidate is considered, reasoning parts are skipped, and
    the first remaining text part wins.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "thought", None):
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str):
            return text
    return None


def _dump_response(response) -> str:
    """Serializes the whole response into the ``full_response`` field.

    The spec fixes this field as a string so that the return type never
    changes. On Spark 4 a caller who wants to query inside it can wrap it in
    ``parse_json`` to get a VARIANT.
    """
    try:
        return response.model_dump_json(exclude_none=True)
    except Exception:  # pylint: disable=broad-except
        logger.debug("Could not serialize the model response.", exc_info=True)
        return EMPTY_RESPONSE
