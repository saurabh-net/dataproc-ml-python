<div align="center">
  <img src="https://cloud.google.com/images/social-icon-google-cloud-1200-630.png" width="120" alt="Google Cloud logo">
  <h1>Dataproc ML</h1>
</div>

[![PyPI version](https://img.shields.io/pypi/v/dataproc-ml)](https://pypi.org/project/dataproc-ml/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Public Preview Disclaimer**
>
> Interfaces and functionality are subject to change. It is not recommended for production-critical applications without thorough testing and understanding of the potential risks.

`Dataproc ML` is a Python library that simplifies distributed ML inference on Google Cloud Dataproc. It provides high-level handlers to run PyTorch and Vertex AI Gemini models at scale using Apache Spark, without the complexity of manual model distribution and batch processing.

## Installation

You can install the library using pip:

```bash
pip install dataproc-ml
```

## Usage Examples

Here are a couple of examples demonstrating how to use the handlers for distributed inference on a Spark DataFrame.

### AI Functions: `ai_generate`

> **Note:** `ai_generate` makes API calls to Vertex AI, which will incur costs.
> Please review the [Vertex AI Generative
> AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing).

`ai_generate` is a Spark column function that calls a Gemini model on every
row. It always returns the same struct, whatever the arguments:

```
STRUCT<result STRING, full_response STRING, status STRING>
```

`result` is the generated text, `full_response` is the whole model response as
JSON, and `status` is `SUCCESS`, `RATE_LIMITED`, `SAFETY_BLOCKED`, or a
description of the error. A row that fails reports it in `status` rather than
failing the query.

```python
from pyspark.sql import SparkSession, functions as F
from google.cloud.dataproc_ml.sql import ai_generate

spark = SparkSession.builder.getOrCreate()

df = spark.createDataFrame([("Paris",), ("Tokyo",)], ["city"])

result_df = df.withColumn(
    "generated",
    ai_generate(F.concat(F.lit("Airport code for "), F.col("city"))),
).select("city", "generated.result", "generated.status")

result_df.show()
# +-----+------+-------+
# | city|result| status|
# +-----+------+-------+
# |Paris|   CDG|SUCCESS|
# |Tokyo|   HND|SUCCESS|
# +-----+------+-------+
```

Pass `output_schema` as a Spark DDL string to constrain the model to a shape.
`result` then holds a JSON document matching it, which you can project with
`from_json` when you want typed columns:

```python
schema = "sentiment STRING, score DOUBLE"

df.withColumn(
    "review",
    ai_generate(F.col("text"), output_schema=schema),
).select(F.from_json(F.col("review.result"), schema).alias("parsed"))
```

Model settings are passed as a JSON string matching the Vertex AI
`generateContent` request body, so anything the API supports works without
this library having to know about it:

```python
ai_generate(
    F.col("text"),
    model_params='{"generation_config": {"temperature": 0.0, "max_output_tokens": 512}}',
)
```

The same function can be registered for use from Spark SQL, where the model
settings are ordinary arguments and may be passed by name in any order:

```python
from google.cloud.dataproc_ml.sql import ai_generate_udf

spark.udf.register("ai_generate", ai_generate_udf())

spark.sql("""
    SELECT ai_generate(
        prompt => CONCAT('Airport code for ', city),
        endpoint => 'gemini-3.6-flash',
        output_schema => 'code STRING'
    ).result
    FROM cities
""").show()
```

### Generative AI (Gemini) Model Inference

> **Note:** Using the `GenAiModelHandler` involves making API calls to 
> Vertex AI, which will incur costs. Please review the [Vertex AI Generative 
> AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing).

Use Google's Gemini models to perform generative tasks on your data.
This example uses a prompt template to ask for the capital of countries listed in a Spark DataFrame.

```python
from pyspark.sql import SparkSession
from google.cloud.dataproc_ml.inference import GenAiModelHandler

spark = SparkSession.builder.getOrCreate()

# Create a sample DataFrame
data = [("USA",), ("France",), ("Japan",)]
input_df = spark.createDataFrame(data, ["country"])

# The handler will automatically use the 'country' column
result_df = (
    GenAiModelHandler()
    .prompt("What is the capital of {country}?")
    .output_col("capital_city")
    .transform(input_df)
)

result_df.show()
# +-------+----------------+
# |country|capital_city    |
# +-------+----------------+
# |USA    |Washington, D.C.|
# |France |Paris           |
# |Japan  |Tokyo           |
# +-------+----------------+
```

### PyTorch Model Inference

Run distributed inference using a pre-trained PyTorch model stored in Google Cloud Storage.
This example assumes you have a Spark DataFrame `input_df` with a column named `features` containing image tensors or other numerical data.

```python
from pyspark.sql import SparkSession
from google.cloud.dataproc_ml.inference import PyTorchModelHandler

spark = SparkSession.builder.getOrCreate()

data = [([0.1, 0.2, 0.3],), ([0.4, 0.5, 0.6],), ([0.7, 0.8, 0.9],)]
input_df = spark.createDataFrame(data, ["features"])

# Path to your saved PyTorch model in GCS
model_gcs_path = "gs://your-bucket/path/to/model.pt"

# Apply the model for inference
result_df = (
    PyTorchModelHandler()
    .model_path(model_gcs_path)
    .input_cols("features")
    .transform(input_df)
)

result_df.show()
# +------------------+--------------------+
# |          features|         predictions|
# +------------------+--------------------+
# |[0.1, 0.2, 0.3]   |[0.543, 0.457]      |
# |[0.4, 0.5, 0.6]   |[0.621, 0.379]      |
# |[0.7, 0.8, 0.9]   |[0.789, 0.211]      |
# +------------------+--------------------+
```

## Documentation

For more detailed information on the available handlers and their configurations,
please refer to our official [documentation](https://dataproc-ml.readthedocs.io/).

## Contributing

Contributions are welcome! Please see contributing.md for details on how to 
set up your development environment, run linters/tests, etc.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.
