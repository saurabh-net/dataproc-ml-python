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

"""Skips for tests that need a Spark feature the running version lacks.

The library supports Spark 3.5 and later, but a few tests are written against
features that only exist in Spark 4: ``StructType.fromDDL``, used as an oracle
for the DDL parser, and the VARIANT functions, used to show that
``full_response`` can be queried. Neither is used by the library itself, so
they gate the test rather than the feature.
"""

import unittest

import pyspark

_MAJOR = int(pyspark.__version__.split(".", maxsplit=1)[0])

#: True when the running PySpark is 4.0 or later.
IS_SPARK_4 = _MAJOR >= 4

#: Skips a test that needs an API added in PySpark 4.
requires_spark_4 = unittest.skipUnless(
    IS_SPARK_4,
    f"needs PySpark 4 (running {pyspark.__version__})",
)
