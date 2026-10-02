# GTE Embeddings & Vector Search on Databricks

## Overview

This notebook demonstrates a complete end-to-end workflow for generating text embeddings with the **GTE Large (En)** foundation model and building a **Vector Search index** for real-time semantic search on Databricks.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Source Table                             │
│   dbacademy.default.documents_for_vector_search              │
│   (id, title, text)                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │  Delta Sync (TRIGGERED)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Vector Search Index                        │
│   dbacademy.default.documents_vector_index                  │
│                                                              │
│   ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│   │  text column│───▶│  GTE embedding    │───▶│  1024-dim  │  │
│   │  (source)   │    │  (auto-generated) │    │  vectors   │  │
│   └─────────────┘    └──────────────────┘    └────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │  ANN Similarity Search
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Vector Search Endpoint                          │
│              vector_search_demo_endpoint                     │
│                                                              │
│   Query: "How to deploy ML models?"                         │
│     │                                                        │
│     ▼  GTE embeds the query → compares vectors → ranks       │
│                                                              │
│   Result: Model Serving (0.60), MLflow (0.59), Databricks   │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

| Requirement | Details |
|---|---|
| Databricks workspace | With Foundation Model APIs enabled |
| Embedding endpoint | `databricks-gte-large-en` (pre-configured, pay-per-token) |
| Unity Catalog schema | Permission to create tables (e.g., `dbacademy.default`) |
| Compute | Serverless or interactive cluster |
| Python packages | `mlflow`, `databricks-sdk`, `numpy` (pre-installed on Databricks) |

## Notebook Contents

| Part | Steps | Description |
|------|-------|-------------|
| **Part I: Querying GTE Embeddings** | Steps 1–4 | Three methods to generate embeddings (MLflow SDK, Databricks SDK, SQL) + cosine similarity verification |
| **Part II: Vector Search** | Steps 5–9 | Create source table, Vector Search endpoint, Delta Sync Index, and run semantic search queries |

## Key Components Created

| Component | Name | Type |
|---|---|---|
| Source table | `dbacademy.default.documents_for_vector_search` | Delta table with 10 sample documents |
| Vector Search endpoint | `vector_search_demo_endpoint` | STANDARD endpoint for similarity search |
| Vector Search index | `dbacademy.default.documents_vector_index` | Delta Sync Index with auto-embeddings |
| Embedding model | `databricks-gte-large-en` | Pay-per-token Foundation Model API (1024-dim, 8192 token window) |

## Results

### Cosine Similarity (Step 4)

| Text Pair | Similarity |
|---|---|
| Databricks vs Apache Spark (both tech) | **0.832** |
| Databricks vs Pizza (tech vs food) | **0.537** |
| Apache Spark vs Pizza (tech vs food) | **0.506** |

### Semantic Search (Step 9)

| Query | Top Result | Score |
|---|---|---|
| "data analytics platforms" | Databricks Platform | 0.610 |
| "popular Italian dish" | Italian Food | 0.565 |
| "deploy machine learning models" | Model Serving | 0.596 |

## How to Run

1. Import the notebook into your Databricks workspace
2. Attach it to a Databricks cluster or serverless compute
3. Run cells top to bottom
4. Steps 1–4 run instantly (embedding queries)
5. Steps 5–9 take ~10 minutes (Vector Search endpoint and index provisioning)
6. Modify the source table or queries to experiment with your own data

## Cleanup

The Summary cell at the end of the notebook includes optional cleanup code to delete the index, endpoint, and source table.

## Resources

- [Databricks Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/)
- [Databricks Vector Search](https://docs.databricks.com/aws/en/ai-search/)
- [GTE Large (En) Model](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/supported-models/)
- [ai_query SQL Function](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query/)
