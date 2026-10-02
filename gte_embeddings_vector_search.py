# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,README
# MAGIC %md
# MAGIC # 📖 README: GTE Embeddings & Vector Search on Databricks
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook demonstrates a complete end-to-end workflow for generating text embeddings with the **GTE Large (En)** foundation model and building a **Vector Search index** for real-time semantic search on Databricks.
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                     Source Table                             │
# MAGIC │   dbacademy.default.documents_for_vector_search              │
# MAGIC │   (id, title, text)                                          │
# MAGIC └──────────────────────────┬──────────────────────────────────┘
# MAGIC                            │
# MAGIC                            │  Delta Sync (TRIGGERED)
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                   Vector Search Index                        │
# MAGIC │   dbacademy.default.documents_vector_index                  │
# MAGIC │                                                              │
# MAGIC │   ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
# MAGIC │   │  text column│───▶│  GTE embedding    │───▶│  1024-dim  │  │
# MAGIC │   │  (source)   │    │  (auto-generated) │    │  vectors   │  │
# MAGIC │   └─────────────┘    └──────────────────┘    └────────────┘  │
# MAGIC └──────────────────────────┬──────────────────────────────────┘
# MAGIC                            │
# MAGIC                            │  ANN Similarity Search
# MAGIC                            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │              Vector Search Endpoint                          │
# MAGIC │              vector_search_demo_endpoint                     │
# MAGIC │                                                              │
# MAGIC │   Query: "How to deploy ML models?"                         │
# MAGIC │     │                                                        │
# MAGIC │     ▼  GTE embeds the query → compares vectors → ranks       │
# MAGIC │                                                              │
# MAGIC │   Result: Model Serving (0.60), MLflow (0.59), Databricks   │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC | Requirement | Details |
# MAGIC |---|---|
# MAGIC | Databricks workspace | With Foundation Model APIs enabled |
# MAGIC | Embedding endpoint | `databricks-gte-large-en` (pre-configured, pay-per-token) |
# MAGIC | Unity Catalog schema | Permission to create tables (e.g., `dbacademy.default`) |
# MAGIC | Compute | Serverless or interactive cluster |
# MAGIC | Python packages | `mlflow`, `databricks-sdk`, `numpy` (pre-installed on Databricks) |
# MAGIC
# MAGIC ## Notebook Contents
# MAGIC
# MAGIC | Part | Steps | Description |
# MAGIC |------|-------|-------------|
# MAGIC | **Part I: Querying GTE Embeddings** | Steps 1–4 | Three methods to generate embeddings (MLflow SDK, Databricks SDK, SQL) + cosine similarity verification |
# MAGIC | **Part II: Vector Search** | Steps 5–9 | Create source table, Vector Search endpoint, Delta Sync Index, and run semantic search queries |
# MAGIC
# MAGIC ## Key Components Created
# MAGIC
# MAGIC | Component | Name | Type |
# MAGIC |---|---|---|
# MAGIC | Source table | `dbacademy.default.documents_for_vector_search` | Delta table with 10 sample documents |
# MAGIC | Vector Search endpoint | `vector_search_demo_endpoint` | STANDARD endpoint for similarity search |
# MAGIC | Vector Search index | `dbacademy.default.documents_vector_index` | Delta Sync Index with auto-embeddings |
# MAGIC | Embedding model | `databricks-gte-large-en` | Pay-per-token Foundation Model API (1024-dim, 8192 token window) |
# MAGIC
# MAGIC ## Results
# MAGIC
# MAGIC ### Cosine Similarity (Step 4)
# MAGIC
# MAGIC | Text Pair | Similarity |
# MAGIC |---|---|
# MAGIC | Databricks vs Apache Spark (both tech) | **0.832** |
# MAGIC | Databricks vs Pizza (tech vs food) | **0.537** |
# MAGIC | Apache Spark vs Pizza (tech vs food) | **0.506** |
# MAGIC
# MAGIC ### Semantic Search (Step 9)
# MAGIC
# MAGIC | Query | Top Result | Score |
# MAGIC |---|---|---|
# MAGIC | "data analytics platforms" | Databricks Platform | 0.610 |
# MAGIC | "popular Italian dish" | Italian Food | 0.565 |
# MAGIC | "deploy machine learning models" | Model Serving | 0.596 |
# MAGIC
# MAGIC ## How to Run
# MAGIC
# MAGIC 1. Attach the notebook to a Databricks cluster or serverless compute
# MAGIC 2. Run cells top to bottom
# MAGIC 3. Steps 1–4 run instantly (embedding queries)
# MAGIC 4. Steps 5–9 take ~10 minutes (Vector Search endpoint and index provisioning)
# MAGIC 5. Modify the source table or queries to experiment with your own data
# MAGIC
# MAGIC ## Cleanup
# MAGIC
# MAGIC The Summary cell at the end includes optional cleanup code to delete the index, endpoint, and source table.
# MAGIC
# MAGIC ## Resources
# MAGIC
# MAGIC - [Databricks Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/)
# MAGIC - [Databricks Vector Search](https://docs.databricks.com/aws/en/ai-search/)
# MAGIC - [GTE Large (En) Model](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models/)
# MAGIC - [ai_query SQL Function](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query/)

# COMMAND ----------

# DBTITLE 1,Title
# MAGIC %md
# MAGIC # GTE Embeddings & Vector Search on Databricks
# MAGIC
# MAGIC This notebook demonstrates an end-to-end workflow for:
# MAGIC
# MAGIC 1. **Querying the `databricks-gte-large-en` Foundation Model API** to generate 1024-dimension text embeddings using three different methods (MLflow SDK, Databricks SDK, and SQL)
# MAGIC 2. **Comparing embeddings** using cosine similarity to verify semantic relevance
# MAGIC 3. **Creating a Vector Search index** backed by GTE embeddings for real-time semantic search
# MAGIC 4. **Querying the Vector Search index** with natural language questions
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks workspace with Foundation Model APIs enabled
# MAGIC - The `databricks-gte-large-en` pay-per-token endpoint (pre-configured, no setup needed)
# MAGIC - Permission to create tables in a Unity Catalog schema

# COMMAND ----------

# DBTITLE 1,Part I Header + Step 1
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part I: Querying GTE Embeddings
# MAGIC
# MAGIC Before building a Vector Search index, let's first understand how to generate embeddings using the `databricks-gte-large-en` Foundation Model API. We'll explore three different methods and verify the embeddings capture semantic meaning.
# MAGIC
# MAGIC ## Step 1: Query GTE Embeddings with MLflow Deployments SDK
# MAGIC
# MAGIC The simplest way to query the `databricks-gte-large-en` endpoint is through the **MLflow Deployments SDK**. This endpoint is a pre-configured pay-per-token Foundation Model API — no endpoint creation needed.
# MAGIC
# MAGIC The GTE Large model maps any text to a **1024-dimension embedding vector** with an 8192-token context window, suitable for retrieval, classification, clustering, and semantic search.

# COMMAND ----------

# DBTITLE 1,Query with MLflow Deployments SDK
import mlflow.deployments

# Create a deployment client for Databricks
client = mlflow.deployments.get_deploy_client("databricks")

# Query the GTE Large endpoint with a single text
response = client.predict(
    endpoint="databricks-gte-large-en",
    inputs={"input": "Databricks is a unified analytics platform for data engineering and machine learning."},
)

# Extract the embedding vector
embedding = response["data"][0]["embedding"]
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# COMMAND ----------

# DBTITLE 1,Step 2 Intro
# MAGIC %md
# MAGIC ## Step 2: Query GTE Embeddings with Databricks SDK
# MAGIC
# MAGIC The **Databricks SDK for Python** provides an alternative way to query the same endpoint. The key difference from the MLflow SDK is the response format — the SDK returns a `QueryEndpointResponse` object (access via `.data[0].embedding`) instead of a dictionary (access via `["data"][0]["embedding"]`).
# MAGIC
# MAGIC For embeddings models, the `input` parameter must be a **string or list of strings**, not a dictionary.

# COMMAND ----------

# DBTITLE 1,Query with Databricks SDK
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Query the GTE Large endpoint using the Databricks SDK
# For embeddings models, 'input' must be a string or list of strings
response = w.serving_endpoints.query(
    name="databricks-gte-large-en",
    input=["Machine learning models can be served at scale on Databricks."],
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# COMMAND ----------

# DBTITLE 1,Step 3 Intro
# MAGIC %md
# MAGIC ## Step 3: Query GTE Embeddings with SQL `ai_query`
# MAGIC
# MAGIC You can also generate embeddings **directly in SQL** using the built-in `ai_query()` function. This is useful for embedding text in data pipelines, batch processing, or when working with SQL warehouses.
# MAGIC
# MAGIC The function takes two arguments:
# MAGIC - The endpoint name (`'databricks-gte-large-en'`)
# MAGIC - The text to embed
# MAGIC
# MAGIC It returns an `ARRAY<DOUBLE>` containing the 1024-dimension embedding vector.

# COMMAND ----------

# DBTITLE 1,Query with SQL ai_query
# MAGIC %sql
# MAGIC -- Use the ai_query SQL function to generate embeddings directly in SQL
# MAGIC -- ai_query() returns an ARRAY<DOUBLE> containing the 1024-dimension embedding vector
# MAGIC SELECT ai_query(
# MAGIC   'databricks-gte-large-en',
# MAGIC   'Databricks Unity Catalog provides unified governance for data and AI assets.'
# MAGIC ) AS embedding;

# COMMAND ----------

# DBTITLE 1,Step 4 Intro
# MAGIC %md
# MAGIC ## Step 4: Compare Embeddings with Cosine Similarity
# MAGIC
# MAGIC Now let's verify that the embeddings make **semantic sense**. We'll embed three texts — two about technology and one about food — and compute cosine similarity between each pair.
# MAGIC
# MAGIC **Expected result:** Tech texts should have higher similarity with each other than with the food text, because GTE captures semantic meaning, not just keyword overlap.

# COMMAND ----------

# DBTITLE 1,Compare multiple texts with cosine similarity
import numpy as np

# Multiple texts to embed and compare
texts = [
    "Databricks is a unified analytics platform.",
    "Apache Spark is a distributed computing engine.",
    "I love eating pizza on weekends.",
]

response = client.predict(
    endpoint="databricks-gte-large-en",
    inputs={"input": texts},
)

embeddings = [item["embedding"] for item in response["data"]]
print(f"Number of embeddings: {len(embeddings)}")
print(f"Each embedding dimension: {len(embeddings[0])}")

# Compute cosine similarity between pairs
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nCosine Similarity:")
print(f"  Text 0 vs Text 1 (both about tech): {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
print(f"  Text 0 vs Text 2 (tech vs food):    {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
print(f"  Text 1 vs Text 2 (tech vs food):    {cosine_similarity(embeddings[1], embeddings[2]):.4f}")

# COMMAND ----------

# DBTITLE 1,Part II Header
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Part II: Vector Search with GTE Embeddings
# MAGIC
# MAGIC Now we'll use the GTE embeddings to power a **Vector Search index** for real-time semantic search. This involves:
# MAGIC
# MAGIC 1. Creating a source table with documents to index
# MAGIC 2. Creating a Vector Search endpoint (the compute that powers similarity search)
# MAGIC 3. Creating a Delta Sync Index that auto-embeds the text column using `databricks-gte-large-en`
# MAGIC 4. Querying the index with natural language questions
# MAGIC
# MAGIC ## Step 5: Create a Source Table
# MAGIC
# MAGIC We'll create a table with 10 sample documents — 7 about Databricks technologies and 3 about food. This mix will let us verify that semantic search correctly distinguishes between topics.

# COMMAND ----------

# DBTITLE 1,Create source table
# MAGIC %sql
# MAGIC -- Create a source table with sample documents for Vector Search
# MAGIC CREATE OR REPLACE TABLE dbacademy.default.documents_for_vector_search (
# MAGIC   id BIGINT,
# MAGIC   title STRING,
# MAGIC   text STRING
# MAGIC );
# MAGIC
# MAGIC INSERT INTO dbacademy.default.documents_for_vector_search VALUES
# MAGIC   (1, 'Databricks Platform', 'Databricks is a unified analytics platform for data engineering, data science, and machine learning. It provides a collaborative environment for processing and analyzing large datasets using Apache Spark.'),
# MAGIC   (2, 'Apache Spark', 'Apache Spark is a distributed computing engine for large-scale data processing. It supports batch processing, streaming, machine learning, and graph computation.'),
# MAGIC   (3, 'Delta Lake', 'Delta Lake is an open-source storage layer that brings ACID transactions to Apache Spark and big data workloads. It provides reliability, performance, and lifecycle management for data lakes.'),
# MAGIC   (4, 'Unity Catalog', 'Unity Catalog is a unified governance solution for data and AI assets on Databricks. It provides centralized access control, auditing, lineage, and data discovery across all workspaces.'),
# MAGIC   (5, 'MLflow', 'MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It supports experiment tracking, model packaging, model registry, and deployment.'),
# MAGIC   (6, 'Model Serving', 'Databricks Model Serving provides real-time inference for machine learning models. It supports autoscaling, scale-to-zero, and GPU workloads for production deployment.'),
# MAGIC   (7, 'Vector Search', 'Databricks Vector Search is a similarity search engine that enables fast and accurate retrieval of relevant data using vector embeddings. It integrates with Foundation Model APIs for automatic embedding generation.'),
# MAGIC   (8, 'Italian Food', 'Pizza is a popular Italian dish made with a flatbread base topped with tomato sauce, cheese, and various ingredients. It is enjoyed worldwide in many variations.'),
# MAGIC   (9, 'French Cuisine', 'Croissants are a buttery, flaky pastry of Austrian origin but most associated with France. They are a staple of French breakfast and bakery culture.'),
# MAGIC   (10, 'Japanese Cooking', 'Sushi is a traditional Japanese dish of prepared vinegared rice accompanied by seafood, vegetables, and occasionally tropical fruits. It is an art form in Japanese cuisine.');
# MAGIC
# MAGIC SELECT * FROM dbacademy.default.documents_for_vector_search;

# COMMAND ----------

# DBTITLE 1,Step 6 Intro
# MAGIC %md
# MAGIC ## Step 6: Create a Vector Search Endpoint
# MAGIC
# MAGIC A **Vector Search endpoint** is the compute resource that powers similarity search queries. It hosts the index and handles ANN (Approximate Nearest Neighbor) searches.
# MAGIC
# MAGIC We use the `STANDARD` endpoint type, which is suitable for most workloads. Endpoint creation takes a few minutes to provision.

# COMMAND ----------

# DBTITLE 1,Create Vector Search endpoint
from databricks.sdk.service.vectorsearch import EndpointType

endpoint_name = "vector_search_demo_endpoint"

# Create the Vector Search endpoint
print(f"Creating Vector Search endpoint: {endpoint_name}...")
w.vector_search_endpoints.create_endpoint(
    name=endpoint_name,
    endpoint_type=EndpointType.STANDARD,
)
print("Endpoint creation initiated.")

# Wait for the endpoint to come online
import time
print("Waiting for endpoint to be online...")
for i in range(60):
    endpoints = list(w.vector_search_endpoints.list_endpoints())
    for ep in endpoints:
        if ep.name == endpoint_name:
            state = ep.endpoint_status.state
            print(f"  Attempt {i+1}: state={state}")
            if state.name == "ONLINE":
                print("✅ Endpoint is online!")
                break
    else:
        time.sleep(10)
        continue
    break

# COMMAND ----------

# DBTITLE 1,Step 7 Intro
# MAGIC %md
# MAGIC ## Step 7: Create the Vector Search Index
# MAGIC
# MAGIC We create a **Delta Sync Index** that automatically:
# MAGIC 1. Reads the `text` column from our source table
# MAGIC 2. Sends it to the `databricks-gte-large-en` endpoint for embedding
# MAGIC 3. Stores the 1024-dimension vectors in the index
# MAGIC 4. Enables similarity search on the embedded content
# MAGIC
# MAGIC We use `TRIGGERED` sync mode (manual sync) instead of `CONTINUOUS` (auto-sync on every change). For production, use `CONTINUOUS` so the index updates automatically when the source table changes.

# COMMAND ----------

# DBTITLE 1,Create Vector Search index
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
)

index_name = "dbacademy.default.documents_vector_index"

# Create a Delta Sync Index with GTE embeddings
print(f"Creating Vector Search index: {index_name}...")
index = w.vector_search_indexes.create_index(
    name=index_name,
    endpoint_name=endpoint_name,
    primary_key="id",
    index_type=VectorIndexType.DELTA_SYNC,
    delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
        source_table="dbacademy.default.documents_for_vector_search",
        pipeline_type=PipelineType.TRIGGERED,
        embedding_source_columns=[
            EmbeddingSourceColumn(
                name="text",
                embedding_model_endpoint_name="databricks-gte-large-en",
            )
        ],
    ),
)

print(f"Index created! Ready: {index.status.ready}")
print(f"Message: {index.status.message}")

# COMMAND ----------

# DBTITLE 1,Step 8 Intro
# MAGIC %md
# MAGIC ## Step 8: Wait for Index to Be Ready
# MAGIC
# MAGIC The index needs time to provision pipeline resources and sync the initial data. The GTE endpoint embeds each document, and the index stores the vectors for fast similarity search.
# MAGIC
# MAGIC This typically takes 5–10 minutes for the first index on a new endpoint.

# COMMAND ----------

# DBTITLE 1,Wait for index readiness
# Wait for the index to be ready and sync initial data
print("Waiting for index to be ready...")
for i in range(60):
    idx = w.vector_search_indexes.get_index(index_name=index_name)
    ready = idx.status.ready
    rows = idx.status.indexed_row_count
    msg = idx.status.message
    print(f"  Attempt {i+1}: ready={ready}, rows={rows}, msg={msg}")
    if ready:
        print("✅ Index is ready with", rows, "documents indexed!")
        break
    time.sleep(15)
else:
    print("⚠️ Timed out waiting. Check the index status in the UI.")

# COMMAND ----------

# DBTITLE 1,Step 9 Intro
# MAGIC %md
# MAGIC ## Step 9: Query the Vector Search Index (Semantic Search)
# MAGIC
# MAGIC Now the fun part — we can search the index using **natural language questions**. The Vector Search endpoint:
# MAGIC 1. Embeds the query text using the same GTE model
# MAGIC 2. Finds the most similar documents by comparing embedding vectors
# MAGIC 3. Returns ranked results with similarity scores
# MAGIC
# MAGIC We'll run three queries that test different semantic domains:
# MAGIC - A tech question → should return tech documents
# MAGIC - A food question → should return food documents
# MAGIC - A deployment question → should return Model Serving and MLflow docs

# COMMAND ----------

# DBTITLE 1,Query Vector Search index
# Query the Vector Search index with natural language questions
queries = [
    "Tell me about data analytics platforms",
    "What is a popular Italian dish?",
    "How to deploy machine learning models?",
]

print("=== Semantic Search Results ===\n")
for query in queries:
    print(f"Query: '{query}'")
    print("-" * 60)
    
    result = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["id", "title", "text"],
        query_text=query,
        num_results=3,
    )
    
    # Extract column names from the manifest
    columns = [c.name for c in result.manifest.columns]
    for row in result.result.data_array:
        doc_id = int(row[columns.index("id")])
        title = row[columns.index("title")]
        score = row[columns.index("score")]
        print(f"  Score: {score:.4f} | ID: {doc_id} | {title}")
    print()

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # Summary
# MAGIC
# MAGIC ## What We Built
# MAGIC
# MAGIC | Component | Name | Purpose |
# MAGIC |-----------|------|---------|
# MAGIC | Source Table | `dbacademy.default.documents_for_vector_search` | 10 sample documents (tech + food) |
# MAGIC | Vector Search Endpoint | `vector_search_demo_endpoint` | Compute for similarity search |
# MAGIC | Vector Search Index | `dbacademy.default.documents_vector_index` | Delta Sync Index with GTE embeddings |
# MAGIC | Embedding Model | `databricks-gte-large-en` | Pay-per-token Foundation Model API |
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC 1. The `databricks-gte-large-en` endpoint is **pre-configured** — no setup needed, just query it
# MAGIC 2. GTE produces **1024-dimension embeddings** that capture semantic meaning
# MAGIC 3. Cosine similarity confirms tech texts score higher with each other (~0.83) than with food texts (~0.52)
# MAGIC 4. Vector Search auto-embeds the `text` column using the GTE endpoint — no manual embedding needed
# MAGIC 5. Semantic search correctly returns relevant documents regardless of keyword overlap
# MAGIC
# MAGIC ## To Add More Documents
# MAGIC
# MAGIC ```python
# MAGIC # Insert new rows into the source table, then trigger a sync
# MAGIC spark.sql("INSERT INTO dbacademy.default.documents_for_vector_search VALUES (11, 'New Doc', 'text here')")
# MAGIC w.vector_search_indexes.sync_index(index_name=index_name)
# MAGIC ```
# MAGIC
# MAGIC ## Cleanup (Optional)
# MAGIC
# MAGIC ```python
# MAGIC # Delete the index
# MAGIC w.vector_search_indexes.delete_index(index_name=index_name)
# MAGIC # Delete the endpoint
# MAGIC w.vector_search_endpoints.delete_endpoint(name=endpoint_name)
# MAGIC # Drop the source table
# MAGIC spark.sql("DROP TABLE IF EXISTS dbacademy.default.documents_for_vector_search")
# MAGIC ```