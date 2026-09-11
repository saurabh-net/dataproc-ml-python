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
import json
import logging
import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import pandas as pd
from pyspark.sql import Column
from pyspark.sql import functions as sf
from pyspark.sql.pandas.functions import pandas_udf
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

from google.cloud.dataproc_ml.sql import _client
from google.cloud.dataproc_ml.sql import _model_params
from google.cloud.dataproc_ml.sql import _output_schema as _output_schema_util
from google.cloud.dataproc_ml.sql import _retry

try:
    # Spark 4 models semi-structured data as VARIANT, which is the closest
    # equivalent of the JSON type that BigQuery's AI functions return.
    from pyspark.sql.types import VariantType, VariantVal  # pylint: disable=ungrouped-imports

    _VARIANT_SUPPORTED = True
except ImportError:  # pragma: no cover - depends on the Spark version
    _VARIANT_SUPPORTED = False


logger = logging.getLogger(__name__)

#: The model used when no endpoint is given.
DEFAULT_ENDPOINT = "gemini-3.6-flash"

# Fixed request settings. BigQuery does not expose these either; they are
# constants here so that the function behaves sensibly on a large cluster
# without asking the caller to tune anything.
_MAX_CONCURRENT_REQUESTS = 5
_MAX_ATTEMPTS = 5
_BASE_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 60.0

_RESULT_FIELD = "result"
_FULL_RESPONSE_FIELD = "full_response"
_STATUS_FIELD = "status"

PromptArg = Union[Column, str, Sequence[Union[Column, str]]]


def ai_generate(
    prompt: PromptArg,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model_params: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Union[str, StructType]] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    _generate_content=None,
) -> Column:
    """Generates text for every row using a Gemini model.

    This is the Spark counterpart of BigQuery's ``AI.GENERATE`` function and
    returns the same struct, so queries can be moved between the two engines
    with little change. It takes the same arguments, except that the Vertex AI
    call is authenticated with Application Default Credentials rather than a
    BigQuery connection, so ``project`` and ``location`` take the place of
    ``connection_id``.

    The prompt follows the usual PySpark convention: a string names a column,
    and literal text must be wrapped in :func:`pyspark.sql.functions.lit`. A
    sequence of parts is concatenated in order, like BigQuery's struct prompt.

    Requests are retried with exponential backoff and are issued a few at a
    time per worker. Like BigQuery, this function does not expose those
    settings.

    .. note::
        Using this function incurs Vertex AI charges. See the
        `Vertex AI Generative AI pricing
        <https://cloud.google.com/vertex-ai/generative-ai/pricing>`_ page.

    Args:
        prompt: The prompt column, the name of a prompt column, or a sequence
            of parts to concatenate.
        endpoint: The Gemini model to call. Defaults to ``gemini-3.6-flash``.
        model_params: Additional request settings, following the Vertex AI
            ``generateContent`` request body without its ``contents`` field,
            for example ``{"generation_config": {"temperature": 0.2}}``.
        output_schema: An optional description of the fields the model should
            return, for example ``"sentiment STRING, score INT64"``. When it
            is set, the ``result`` field is replaced by one field per declared
            field, sorted by name as BigQuery does.
        project: The Google Cloud project billed for the calls. Inferred from
            the environment when not set.
        location: The Vertex AI location. Defaults to ``global``, which is
            where the current Gemini models are served. Set it only when a
            particular region is required, and check that the endpoint is
            available there.

    Returns:
        A column holding a struct. Without ``output_schema`` the fields are
        ``result``, ``full_response`` and ``status``; with it, the declared
        fields replace ``result``. ``full_response`` holds the complete model
        response, typed as VARIANT on Spark 4 and as a JSON string on older
        versions. ``status`` is empty when the row succeeded and holds the
        error message otherwise.

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
    generate = ai_generate_udf(
        endpoint=endpoint,
        model_params=model_params,
        output_schema=output_schema,
        project=project,
        location=location,
        _generate_content=_generate_content,
    )
    return generate(_to_prompt_column(prompt))


def ai_generate_udf(
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model_params: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Union[str, StructType]] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    _generate_content=None,
):
    """Builds a reusable ``AI.GENERATE`` user defined function.

    Use this when the same configuration is applied in several places, or to
    make the function available to Spark SQL:

        >>> spark.udf.register("ai_generate", ai_generate_udf())
        >>> spark.sql("SELECT ai_generate(review).result FROM reviews")

    Args:
        endpoint: See :func:`ai_generate`.
        model_params: See :func:`ai_generate`.
        output_schema: See :func:`ai_generate`.
        project: See :func:`ai_generate`.
        location: See :func:`ai_generate`.
        _generate_content: Internal. Replaces the call to Vertex AI with an
            awaitable of the same shape, so that tests can run the whole Spark
            path without credentials or network access.

    Returns:
        A pandas user defined function taking a single prompt column.

    Raises:
        ValueError: If any argument is invalid. All validation happens here, on
            the driver, so that mistakes surface before a stage is scheduled.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty model name.")

    parsed_schema = (
        _output_schema_util.parse_output_schema(output_schema)
        if output_schema is not None
        else None
    )
    response_schema = (
        _output_schema_util.to_response_schema(parsed_schema)
        if parsed_schema is not None
        else None
    )
    flat_params = _model_params.normalize_model_params(
        model_params, response_schema
    )
    return_type = _build_return_type(parsed_schema)

    def _predict(
        batch_iterator: Iterator[pd.Series],
    ) -> Iterator[pd.DataFrame]:
        """Runs on the executor, once per task."""
        # Resolved once per task and shared with every other task running in
        # this worker process.
        generate_content = _generate_content or _client.get_generate_content(
            project, location
        )
        config = _model_params.build_generation_config(flat_params)

        for batch in batch_iterator:
            rows = _client.run_coroutine(
                _generate_batch(
                    generate_content=generate_content,
                    prompts=batch.tolist(),
                    endpoint=endpoint,
                    config=config,
                    schema=parsed_schema,
                )
            )
            yield pd.DataFrame(
                rows,
                columns=[field.name for field in return_type.fields],
                index=batch.index,
            )

    return pandas_udf(_predict, returnType=return_type)


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
        # Concatenation matches BigQuery, where the parts of a struct prompt
        # are combined in the order they are given.
        return sf.concat(*[column.cast(StringType()) for column in columns])
    raise ValueError(
        "prompt must be a Column, a column name, or a sequence of those, got "
        f"{type(prompt).__name__}."
    )


def _build_return_type(parsed_schema: Optional[StructType]) -> StructType:
    """Builds the struct returned by the function."""
    if parsed_schema is None:
        fields = [StructField(_RESULT_FIELD, StringType())]
    else:
        fields = list(parsed_schema.fields)
        for reserved in (_FULL_RESPONSE_FIELD, _STATUS_FIELD):
            if any(field.name == reserved for field in fields):
                raise ValueError(
                    f"output_schema must not declare a '{reserved}' field; it "
                    "is always added to the result."
                )
    # Every field is nullable: a row that fails reports the failure in status
    # and leaves the generated fields empty.
    fields = [
        StructField(field.name, field.dataType, nullable=True)
        for field in fields
    ]
    fields.append(StructField(_FULL_RESPONSE_FIELD, _full_response_data_type()))
    fields.append(StructField(_STATUS_FIELD, StringType()))
    return StructType(fields)


def _full_response_data_type() -> DataType:
    """Returns the type of the ``full_response`` field.

    BigQuery returns the raw model response as JSON. Spark 4 has an equivalent
    semi-structured type, VARIANT, which allows ``variant_get`` and
    ``to_json`` to be used on the column. On older Spark versions, which have
    no such type, the response is returned as a JSON string instead.
    """
    return VariantType() if _VARIANT_SUPPORTED else StringType()


async def _generate_batch(
    *,
    generate_content,
    prompts: List[Any],
    endpoint: str,
    config,
    schema: Optional[StructType],
) -> List[Dict[str, Any]]:
    """Generates a response for every prompt in an Arrow batch."""
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

    async def _one(prompt: Any) -> Dict[str, Any]:
        async with semaphore:
            return await _generate_one(
                generate_content=generate_content,
                prompt=prompt,
                endpoint=endpoint,
                config=config,
                schema=schema,
            )

    return await asyncio.gather(*[_one(prompt) for prompt in prompts])


async def _generate_one(
    *,
    generate_content,
    prompt: Any,
    endpoint: str,
    config,
    schema: Optional[StructType],
) -> Dict[str, Any]:
    """Generates the response for one row, retrying transient failures."""
    if _is_missing(prompt):
        # A null prompt produces a null result without calling the model.
        return _row(schema, result=None, full_response=None, status="")

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await generate_content(
                model=endpoint, contents=prompt, config=config
            )
            return _to_row(response, schema)
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
                    endpoint,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
                continue
            return _row(
                schema,
                result=None,
                full_response=None,
                status=_retry.to_status_message(e),
            )

    # Unreachable: the loop either returns a row or retries.
    raise AssertionError("The retry loop ended without producing a row.")


def _is_missing(prompt: Any) -> bool:
    """Reports whether a prompt is null.

    Spark nulls arrive as ``None`` or, in a numeric batch, as NaN.
    """
    if prompt is None:
        return True
    if isinstance(prompt, float) and math.isnan(prompt):
        return True
    return not isinstance(prompt, str)


def _to_row(response, schema: Optional[StructType]) -> Dict[str, Any]:
    """Converts a model response into one output row."""
    full_response = _dump_response(response)
    text = _extract_text(response)

    if schema is None:
        return _row(schema, result=text, full_response=full_response, status="")

    if text is None:
        # The call succeeded but produced no usable text, for example because
        # the response was filtered. This is not an error.
        return _row(schema, result=None, full_response=full_response, status="")

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return _row(
            schema,
            result=None,
            full_response=full_response,
            status=_retry.PARSE_FAILURE_STATUS,
        )

    if not isinstance(payload, dict):
        return _row(
            schema,
            result=None,
            full_response=full_response,
            status=_retry.PARSE_FAILURE_STATUS,
        )

    row = {
        field.name: _coerce(payload.get(field.name), field.dataType)
        for field in schema.fields
    }
    row[_FULL_RESPONSE_FIELD] = full_response
    row[_STATUS_FIELD] = ""
    return row


def _row(
    schema: Optional[StructType],
    *,
    result: Optional[str],
    full_response: Any,
    status: str,
) -> Dict[str, Any]:
    """Builds an output row, leaving the generated fields empty if needed."""
    if schema is None:
        row: Dict[str, Any] = {_RESULT_FIELD: result}
    else:
        row = {field.name: None for field in schema.fields}
    row[_FULL_RESPONSE_FIELD] = full_response
    row[_STATUS_FIELD] = status
    return row


def _extract_text(response) -> Optional[str]:
    """Returns the generated text, following BigQuery's extraction rules.

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


def _dump_response(response) -> Any:
    """Serializes the whole response, as BigQuery's ``full_response`` does.

    Args:
        response: The model response.

    Returns:
        A Spark variant value, or a JSON string on Spark versions without the
        variant type.
    """
    try:
        serialized = response.model_dump_json(exclude_none=True)
    except Exception:  # pylint: disable=broad-except
        logger.debug("Could not serialize the model response.", exc_info=True)
        serialized = "{}"
    return _to_full_response(serialized)


def _to_full_response(serialized: str) -> Any:
    """Converts serialized JSON into the value stored in ``full_response``."""
    if not _VARIANT_SUPPORTED:
        return serialized
    try:
        return VariantVal.parseJson(serialized)
    except Exception:  # pylint: disable=broad-except
        logger.debug("Could not encode the response as a variant.")
        return VariantVal.parseJson("{}")


def _coerce(value: Any, data_type: DataType) -> Any:
    """Converts a JSON value to the type declared in the output schema."""
    if value is None:
        return None
    try:
        if isinstance(data_type, StringType):
            return value if isinstance(value, str) else json.dumps(value)
        if isinstance(data_type, BooleanType):
            return bool(value)
        if isinstance(data_type, (LongType, IntegerType)):
            return int(value)
        if isinstance(data_type, (DoubleType, FloatType)):
            return float(value)
        if isinstance(data_type, ArrayType):
            if not isinstance(value, list):
                return None
            return [_coerce(item, data_type.elementType) for item in value]
        if isinstance(data_type, StructType):
            if not isinstance(value, dict):
                return None
            return {
                field.name: _coerce(value.get(field.name), field.dataType)
                for field in data_type.fields
            }
    except (TypeError, ValueError):
        return None
    return None
