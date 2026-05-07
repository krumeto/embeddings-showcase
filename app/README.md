# Streamlit App

This folder contains the local multipage dashboard for the embeddings showcase.

Run it from the repo root:

```bash
uv run streamlit run app/Home.py
```

Pages:

- `Home.py`: artifact status, dataset snapshot, and links to each workflow.
- `pages/1_Vector_Search.py`: semantic search over stored OpenAI and Hugging Face embeddings.
- `pages/2_Clustering.py`: EVōC hierarchical clustering with a Plotly treemap and cluster details.
- `pages/3_Topic_Modelling.py`: Turftopic Topeax topic modelling with algorithm-step plots, topic summaries, and article exploration.
- `pages/4_Classification.py`: multilabel classification using logistic regression on top of the stored embeddings.

The app expects these generated local files:

```text
data/raw/newswire_5000_sample.parquet
data/embeddings/openai_embeddings.npy
data/embeddings/hf_embeddings.npy
```

The generated data and embedding files are intentionally ignored by git.
