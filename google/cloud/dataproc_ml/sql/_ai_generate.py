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
from google.cloud.dataproc_ml.sql import _model_params
from google.cloud.dataproc_ml.sql import _retry

logger = logging.getLogger(__name__)

#: The model used when no endpoint is given.
DEFAULT_ENDPOINT = "gemini-3.6-flash"

# Fixed request settings. These will be configurable in a follow-up PR.
_MAX_CONCURRENT_REQUESTS = 5
_MAX_ATTEMPTS = 5
_BASE_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 60.0

# How many distinct configurations to keep resolved. A query normally uses one,
# supplied as a literal; the cache only grows when the configuration is itself
# computed per row.
_CONFIG_CACHE_SIZE = 64

RESULT_FIELD = "result"
FULL_RESPONSE_FIELD = "full_response"
STATUS_FIELD = "status"

#: The row succeeded. A null prompt also reports success, with a null result.
STATUS_SUCCESS = "SUCCESS"
#: The request was throttled and did not succeed within the retry budget.
STATUS_RATE_LIMITED = _retry.RATE_LIMITED_STATUS
#: The model refused to answer because the prompt or response was blocked.
STATUS_SAFETY_BLOCKED = "SAFETY_BLOCKED"

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

PromptArg = Union[Column, str, Sequence[Union[Column, str]]]
ModelParamsArg = Union[str, Dict[str, Any]]
OutputSchemaArg = Union[str, StructType]


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
    sequence of parts is concatenated in order.

    Requests are retried with exponential backoff and are issued a few at a
    time per worker.

    .. note::
        Using this function incurs Vertex AI charges. See the
        `Vertex AI Generative AI pricing
        <https://cloud.google.com/vertex-ai/generative-ai/pricing>`_ page.

    Args:
        prompt: The prompt column, the name of a prompt column, or a sequence
            of parts to concatenate.
        endpoint: The Gemini model to call, either a name such as
            ``gemini-3.6-flash`` or a fully qualified
            ``projects/.../publishers/google/models/...`` path. Defaults to
            ``gemini-3.6-flash``.
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
        when ``output_schema`` is set. ``full_response`` is the whole model
        response as JSON. ``status`` is ``SUCCESS``, ``RATE_LIMITED``,
        ``SAFETY_BLOCKED``, or a description of the error.

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
    # instead of a column of failed rows.
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
    function is built. They are ordinary arguments, so a query can vary them,
    and they may be omitted or passed by name in any order.

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
        prompt: pd.Series,
        endpoint: pd.Series = None,
        model_params: pd.Series = None,
        output_schema: pd.Series = None,
    ) -> pd.DataFrame:
        """Runs on the executor, once per Arrow batch.

        Spark omits an argument the caller did not supply, which arrives here
        as ``None`` rather than as a column of nulls. That is the only way to
        tell "not given" from "given as NULL", so it is what selects the
        defaults.
        """
        # Cached for the life of the worker process, so this is a lookup
        # rather than a new connection pool per batch.
        generate_content = _generate_content or _client.get_generate_content(
            project, location
        )
        size = len(prompt)
        rows = _client.run_coroutine(
            _generate_batch(
                generate_content=generate_content,
                prompts=prompt.tolist(),
                endpoints=_argument_values(endpoint, size),
                model_params=_argument_values(model_params, size),
                output_schemas=_argument_values(output_schema, size),
            )
        )
        return pd.DataFrame(rows, columns=_RETURN_FIELDS, index=prompt.index)

    return _generate


def _argument_values(argument: Optional[pd.Series], size: int) -> List[Any]:
    """Returns one value per row for an argument that may be absent."""
    if argument is None:
        return [None] * size
    return argument.tolist()


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

    Both arguments are ordinary columns, so in principle they can differ on
    every row. In practice a query supplies literals, and the cache turns the
    repeated work into a single parse per distinct configuration. The result
    is only read, never mutated, so sharing one config across rows and threads
    is safe.

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
    """Builds the single string column that holds each row's prompt.

    Every path casts to STRING. A prompt column of another type would
    otherwise reach the worker as a non-string value and be treated as a null
    prompt, turning a whole column into nulls without any error.
    """
    if isinstance(prompt, Column):
        return prompt.cast(StringType())
    if isinstance(prompt, str):
        return sf.col(prompt).cast(StringType())
    if isinstance(prompt, Sequence):
        parts = list(prompt)
        if not parts:
            raise ValueError("prompt must contain at least one part.")
        columns = [
            part if isinstance(part, Column) else sf.col(part) for part in parts
        ]
        if len(columns) == 1:
            return columns[0].cast(StringType())
        return sf.concat(*[column.cast(StringType()) for column in columns])
    raise ValueError(
        "prompt must be a Column, a column name, or a sequence of those, got "
        f"{type(prompt).__name__}."
    )


async def _generate_batch(
    *,
    generate_content,
    prompts: List[Any],
    endpoints: List[Any],
    model_params: List[Any],
    output_schemas: List[Any],
) -> List[Dict[str, Any]]:
    """Generates a response for every prompt in an Arrow batch."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

    async def _one(index: int) -> Dict[str, Any]:
        async with semaphore:
            return await _generate_one(
                generate_content=generate_content,
                prompt=prompts[index],
                endpoint=endpoints[index],
                model_params=model_params[index],
                output_schema=output_schemas[index],
            )

    return await asyncio.gather(*[_one(i) for i in range(len(prompts))])


async def _generate_one(
    *,
    generate_content,
    prompt: Any,
    endpoint: Any,
    model_params: Any,
    output_schema: Any,
) -> Dict[str, Any]:
    """Generates the response for one row, retrying transient failures."""
    if _is_missing(prompt):
        # A null prompt produces a null result without calling the model.
        return _row(result=None, full_response=None, status=STATUS_SUCCESS)

    # ``ai_generate`` casts its prompt column to STRING on the driver, but the
    # function returned by ``ai_generate_udf`` can also be registered and
    # called straight from Spark SQL, where no such cast is inserted. Render
    # the value here so that a non-string column, say an id, still produces a
    # prompt rather than being dropped.
    text = _to_text(prompt)
    model = _argument_text(endpoint) or DEFAULT_ENDPOINT

    try:
        config = _resolve_config(
            _argument_text(model_params), _argument_text(output_schema)
        )
    except ValueError as e:
        # Only the driver-side entry point can reject these before the query
        # runs. Called from SQL the arguments are columns, so a bad value has
        # to be reported per row like any other failure.
        return _row(result=None, full_response=None, status=str(e))

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await generate_content(
                model=model, contents=text, config=config
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
            return _row(
                result=None,
                full_response=None,
                status=_retry.to_status(e),
            )

    # Unreachable: the loop either returns a row or retries.
    raise AssertionError("The retry loop ended without producing a row.")


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
    """Normalizes one row's value of a string argument.

    An argument that was not supplied, was supplied as NULL, or was supplied
    as an empty string all mean the same thing: use the default.
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

    if _is_safety_blocked(response):
        return _row(
            result=None,
            full_response=full_response,
            status=STATUS_SAFETY_BLOCKED,
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
    full_response: Optional[str],
    status: str,
) -> Dict[str, Any]:
    """Builds an output row."""
    return {
        RESULT_FIELD: result,
        FULL_RESPONSE_FIELD: full_response,
        STATUS_FIELD: status,
    }


def _is_safety_blocked(response) -> bool:
    """Reports whether the model withheld the answer rather than generating it.

    A block is reported either on the prompt, before anything is generated, or
    as the finish reason of the candidate.
    """
    feedback = getattr(response, "prompt_feedback", None)
    if getattr(feedback, "block_reason", None):
        return True

    for candidate in getattr(response, "candidates", None) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is None:
            continue
        # The client library returns an enum, but a plain string is accepted
        # too so that a test or a future version needs no special handling.
        name = getattr(reason, "name", None) or str(reason)
        if name.upper() in _BLOCKED_FINISH_REASONS:
            return True
    return False


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
        return "{}"
