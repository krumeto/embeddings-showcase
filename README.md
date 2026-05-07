# embeddings-showcase

A learning-first repo to showcase what dense embeddings can do. Embeddings are cheap, flexible, intuitive, and useful across search, clustering, topic exploration, and classification.

The goal is to keep the code readable and interactive rather than production-heavy. For example, this project intentionally starts with NumPy arrays and parquet files instead of a full vector database.

## Planned showcase areas

1. Simple vector search
2. Clustering
3. Topic modelling
4. Classification

## Dataset

This project uses the Hugging Face dataset [`dell-research-harvard/newswire`](https://huggingface.co/datasets/dell-research-harvard/newswire):

```
from datasets import load_dataset

load_dataset(
    "dell-research-harvard/newswire",
    data_files=["1972_data_clean.json", "1973_data_clean.json"]
)
```

The default preparation script currently samples labelled articles from `1973_data_clean.json`.

## Setup

This repo uses Python 3.12 and `uv`.

```bash
uv sync
```

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Environment variables:

```bash
OPENAI_KEY=XXX
HF_KEY=XXX
DEFAULT_OPENAI_MODEL=text-embedding-3-large
DEFAULT_HF_MODEL=ibm-granite/granite-embedding-97m-multilingual-r2
```

## Prepare local data

Create a local parquet sample:

```bash
uv run python scripts/prepare_dataset.py
```

This writes `data/raw/newswire_5000_sample.parquet` by default.

Generate embeddings after the parquet file exists:

```bash
uv run python scripts/embed_openai.py --limit 100 --output data/embeddings/openai_test_embeddings.npy
uv run python scripts/embed_hf.py --limit 100 --output data/embeddings/hf_test_embeddings.npy
```

The embedding scripts call remote APIs and require valid keys. The dataset preparation script downloads from Hugging Face but does not call paid embedding APIs.

## Structure

```text
src/embeddings_showcase/  reusable data, embedding, search, clustering, and projection helpers
scripts/                  runnable preparation scripts
app/                      reserved for a future Streamlit dashboard
data/                     local generated artifacts, ignored by git
notebooks/                optional scratch exploration
```

## Streamlit

The Streamlit dashboard is intentionally not implemented yet. The current repository state prepares the data and embedding layers the dashboard will use later.
