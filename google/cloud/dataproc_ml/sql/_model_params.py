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

"""Translation of the ``model_params`` option into a generation config.

``model_params`` follows the Vertex AI ``generateContent`` request body, minus
the ``contents`` field which is supplied by the prompt. That body nests the
generation settings under ``generation_config`` whereas the client library
takes them as a flat configuration object, so this module flattens the request
body and validates it.
"""

import re
from typing import Any, Dict, Optional

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")

# Keys whose contents are merged into the top level of the generation config.
_GENERATION_CONFIG_KEYS = frozenset({"generation_config", "generationconfig"})

_CONTENTS_ERROR = (
    "The model_params is invalid and should not contain contents. The prompt "
    "argument supplies the contents of the request."
)

_BOTH_SCHEMAS_ERROR = (
    "The LLM output schema can only be set either in the model_params or the "
    "output_schema, not both."
)


def _to_snake_case(key: str) -> str:
    """Converts a camelCase request body key to its snake_case equivalent."""
    return _CAMEL_BOUNDARY.sub("_", key).lower()


def normalize_model_params(
    model_params: Optional[Dict[str, Any]],
    response_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flattens ``model_params`` into generation config keyword arguments.

    Both camelCase and snake_case keys are accepted, matching the behavior of
    the JSON request body.

    Args:
        model_params: The user supplied request body, or None.
        response_schema: A response schema derived from ``output_schema``, or
            None when the function returns free form text.

    Returns:
        A dictionary of keyword arguments for the generation config.

    Raises:
        ValueError: If the parameters contain ``contents``, declare a response
            schema that conflicts with ``output_schema``, or do not describe a
            valid request body.
    """
    params = model_params or {}
    if not isinstance(params, dict):
        raise ValueError(
            "model_params must be a dictionary matching the generateContent "
            f"request body, got {type(params).__name__}."
        )

    flat: Dict[str, Any] = {}
    for key, value in params.items():
        snake_key = _to_snake_case(str(key))
        if snake_key == "contents":
            raise ValueError(_CONTENTS_ERROR)
        if snake_key in _GENERATION_CONFIG_KEYS:
            if not isinstance(value, dict):
                raise ValueError(
                    "model_params.generation_config must be a dictionary, got "
                    f"{type(value).__name__}."
                )
            for nested_key, nested_value in value.items():
                flat[_to_snake_case(str(nested_key))] = nested_value
        else:
            flat[snake_key] = value

    if response_schema is not None:
        if flat.get("response_schema") or flat.get("response_json_schema"):
            raise ValueError(_BOTH_SCHEMAS_ERROR)
        flat["response_mime_type"] = "application/json"
        flat["response_schema"] = response_schema

    # Building the config here surfaces invalid parameters on the driver,
    # before any executor work is scheduled.
    build_generation_config(flat)
    return flat


def build_generation_config(flat_params: Dict[str, Any]):
    """Builds the client library generation config.

    Args:
        flat_params: Keyword arguments produced by
            :func:`normalize_model_params`.

    Returns:
        A ``google.genai.types.GenerateContentConfig``.

    Raises:
        ValueError: If the parameters do not describe a valid request body.
    """
    # Imported lazily to keep the Vertex AI dependency off the import path of
    # the package itself.
    from google.genai import types  # pylint: disable=import-outside-toplevel

    try:
        return types.GenerateContentConfig(**flat_params)
    except Exception as e:
        raise ValueError(
            "The model_params is invalid and does not conform to the "
            "generateContent request body. See "
            "https://cloud.google.com/vertex-ai/docs/reference/rest/v1/"
            f"projects.locations.publishers.models/generateContent. Cause: {e}"
        ) from e
