# embeddings-showcase

A learning-first repository for exploring what dense embeddings can do.

Embeddings turn text into vectors. Once text is represented as vectors, we can compare articles by meaning, search with natural-language queries, cluster similar stories, build topic views, and create small classifiers.

This repo intentionally favors readable code and local files over production infrastructure. For a 5,000-row teaching dataset, NumPy arrays and parquet files are easier to understand than a vector database.

## What This Project Will Showcase

1. Simple vector search
2. Clustering
3. Topic modelling
4. Classification

The Streamlit dashboard is planned, but it is not implemented yet. The current repo prepares the data and embedding layers first.

## Dataset

The examples use a labelled subset of the Hugging Face dataset [`dell-research-harvard/newswire`](https://huggingface.co/datasets/dell-research-harvard/newswire).

The preparation script loads `1973_data_clean.json`, keeps the article text and selected topic labels, then creates a 5,000-row labelled sample.

Labels used:

```text
antitrust
civil_rights
crime
govt_regulation
labor_movement
politics
protests
```

## Setup

This project uses Python 3.12 and `uv`.

```bash
uv sync
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Fill in the API keys:

```bash
OPENAI_KEY=XXX
HF_KEY=XXX
DEFAULT_OPENAI_MODEL=text-embedding-3-large
DEFAULT_HF_MODEL=ibm-granite/granite-embedding-97m-multilingual-r2
```

## Generate The Raw Sample

Download the Hugging Face dataset subset and write the 5,000-row sample to parquet:

```bash
uv run python scripts/prepare_dataset.py
```

Default output:

```text
data/raw/newswire_5000_sample.parquet
```

Expected size is about 5 MB.

## Test The Embedding APIs

Before embedding all 5,000 rows, test each provider on 6 articles.

OpenAI test:

```bash
uv run python scripts/embed_openai.py \
  --limit 6 \
  --output data/embeddings/openai_test_embeddings.npy \
  --batch-size 6
```

Expected shape:

```text
(6, 3072)
```

Hugging Face test:

```bash
uv run python scripts/embed_hf.py \
  --limit 6 \
  --output data/embeddings/hf_test_embeddings.npy \
  --batch-size 6
```

Expected shape:

```text
(6, 384)
```

## Generate Full Embeddings

Generate OpenAI embeddings for all 5,000 rows:

```bash
uv run python scripts/embed_openai.py \
  --output data/embeddings/openai_embeddings.npy \
  --batch-size 64 \
  --max-chars 8000
```

Expected output:

```text
data/embeddings/openai_embeddings.npy
shape: (5000, 3072)
size: about 59 MB
```

Generate Hugging Face embeddings for all 5,000 rows:

```bash
uv run python scripts/embed_hf.py \
  --output data/embeddings/hf_embeddings.npy \
  --batch-size 4 \
  --max-chars 4000 \
  --timeout 60 \
  --save-each-batch
```

Expected output:

```text
data/embeddings/hf_embeddings.npy
shape: (5000, 384)
size: about 7 MB
```

The Hugging Face run uses smaller batches and incremental saving because the hosted inference endpoint can be slower or less predictable on long article batches.

## Check The Local Artifacts

Verify row counts, shapes, dtypes, and finite values:

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

raw = Path("data/raw/newswire_5000_sample.parquet")
openai_path = Path("data/embeddings/openai_embeddings.npy")
hf_path = Path("data/embeddings/hf_embeddings.npy")

df = pd.read_parquet(raw)
openai = np.load(openai_path)
hf = np.load(hf_path)

print("raw rows:", len(df))
print("openai:", openai.shape, openai.dtype, "finite:", np.isfinite(openai).all())
print("hf:", hf.shape, hf.dtype, "finite:", np.isfinite(hf).all())
PY
```

Expected result:

```text
raw rows: 5000
openai: (5000, 3072) float32 finite: True
hf: (5000, 384) float32 finite: True
```

## Repository Structure

```text
src/embeddings_showcase/  reusable helpers for data, embeddings, similarity, clustering, and projection
scripts/                  runnable data and embedding generation scripts
app/                      reserved for the future Streamlit dashboard
data/raw/                 local generated parquet samples
data/embeddings/          local generated embedding arrays
notebooks/                optional scratch exploration
```

## Git And Data Files

Generated data and embeddings are intentionally ignored by git:

```text
data/raw/newswire_5000_sample.parquet
data/embeddings/*.npy
```

The full generated artifacts are useful locally, but they are binary files and would bloat repo history. Commit the code, scripts, docs, lockfile, and `.gitkeep` files; regenerate the data locally when needed.
