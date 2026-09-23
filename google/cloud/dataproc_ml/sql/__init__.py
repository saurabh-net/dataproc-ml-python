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

"""Declarative AI functions for Spark DataFrames and Spark SQL.

These functions apply Google's Gemini models to a DataFrame column without any
of the usual plumbing: no client setup, no retry loops and no response
parsing. They work on Apache Spark 3.5 and later.

Example:
    >>> from pyspark.sql.functions import col
    >>> from google.cloud.dataproc_ml.sql import ai_generate
    >>>
    >>> summarized = df.withColumn(
    ...     "summary", ai_generate(col("feedback")).result
    ... )

A prompt may include a file, which Vertex AI reads directly:

    >>> from google.cloud.dataproc_ml.sql import ai_generate, file
    >>>
    >>> described = df.withColumn(
    ...     "g", ai_generate([lit("Describe this image."), file(col("uri"))])
    ... )

The same function can be registered for use from Spark SQL, where the call is
``ai_generate(prompt [, endpoint] [, options])`` and the settings travel in
the ``options`` struct:

    >>> from google.cloud.dataproc_ml.sql import ai_generate_udf
    >>>
    >>> spark.udf.register("ai_generate", ai_generate_udf())
    >>> spark.sql('''
    ...     SELECT ai_generate(
    ...         concat('Summarize: ', feedback),
    ...         named_struct('output_schema', 'summary STRING')
    ...     ).result
    ...     FROM feedback_table
    ... ''')

Every call returns ``STRUCT<result STRING, full_response STRING,
status STRING>``. ``status`` is empty when the row succeeded, so the rows that
need attention are collected with ``WHERE g.status <> ''``.
"""

from ._ai_generate import ai_generate
from ._ai_generate import ai_generate_udf
from ._ai_generate import file

__all__ = (
    "ai_generate",
    "ai_generate_udf",
    "file",
)
