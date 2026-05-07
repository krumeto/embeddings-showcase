# embeddings-showcase

A learning-first repository for exploring what dense embeddings can do.

Embeddings turn text into vectors. Once text is represented as vectors, we can compare articles by meaning, search with natural-language queries, cluster similar stories, build topic views, and create small classifiers.

This repo intentionally favors readable code and local files over production infrastructure. For a 5,000-row teaching dataset, NumPy arrays and parquet files are easier to understand than a vector database.

## What This Project Will Showcase

1. Simple vector search
2. EVōC hierarchical clustering
3. Topeax topic modelling
4. Embedding-based classification

The Streamlit dashboard is implemented as a local multipage app. It uses the
stored parquet and NumPy artifacts directly, so the interactive pages can run
without a vector database or remote backend.

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

## Run The Streamlit App

Start the multipage dashboard:

```bash
uv run streamlit run app/Home.py
```

Then open:

```text
http://localhost:8501
```

The app includes:

- **Home:** local artifact status and dataset snapshot.
- **Vector Search:** compare OpenAI and Hugging Face semantic search results.
- **Clustering:** fit EVōC over stored embeddings, inspect broad-to-fine cluster
  layers in a Plotly treemap, and review representative articles.
- **Topic Modelling:** fit Turftopic Topeax with precomputed embeddings, view the
  Topeax algorithm steps, topic map, topic summaries, and all articles assigned
  to a selected topic.
- **Classification:** train multilabel logistic regression classifiers over the
  stored embeddings and compare provider metrics.

The clustering and topic modelling pages cache model fits by their sidebar
settings. First runs can take a little longer because EVōC and Topeax initialize
their numerical stacks.

## Repository Structure

```text
src/embeddings_showcase/  reusable helpers for data, embeddings, similarity, clustering, topics, classification, and projection
scripts/                  runnable data and embedding generation scripts
app/                      Streamlit multipage dashboard
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
