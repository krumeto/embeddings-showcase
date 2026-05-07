from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from embeddings_showcase.embeddings import embed_with_huggingface, embed_with_openai
from embeddings_showcase.similarity import top_k_search


ROOT = Path(__file__).resolve().parents[2]
RAW_SAMPLE_PATH = ROOT / "data" / "raw" / "newswire_5000_sample.parquet"
OPENAI_EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "openai_embeddings.npy"
HF_EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "hf_embeddings.npy"
LABEL_COLUMNS = [
    "antitrust",
    "civil_rights",
    "crime",
    "govt_regulation",
    "labor_movement",
    "politics",
    "protests",
]


st.set_page_config(
    page_title="Vector Search",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_articles(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


@st.cache_data(show_spinner=False)
def cached_query_embedding(provider: str, query: str) -> np.ndarray:
    if provider == "openai":
        return embed_with_openai([query])[0]
    if provider == "hf":
        return embed_with_huggingface([query], timeout=60)[0]
    raise ValueError(f"Unknown provider: {provider}")


def active_labels(row: pd.Series) -> str:
    labels = [label.replace("_", " ") for label in LABEL_COLUMNS if row.get(label, 0)]
    return ", ".join(labels) if labels else "unlabelled"


def article_preview(text: str, max_chars: int = 650) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def render_results(
    title: str,
    query_vector: np.ndarray,
    embedding_matrix: np.ndarray,
    articles: pd.DataFrame,
    k: int,
) -> None:
    st.subheader(title)

    for rank, (row_index, score) in enumerate(
        top_k_search(query_vector, embedding_matrix, k=k),
        start=1,
    ):
        row = articles.iloc[row_index]
        with st.container(border=True):
            st.caption(f"Rank {rank} · similarity {score:.3f} · row {row_index}")
            st.markdown(f"**Labels:** {active_labels(row)}")
            st.write(article_preview(row["article"]))


def render_process_diagram() -> None:
    st.graphviz_chart(
        """
        digraph {
            graph [rankdir=LR, bgcolor="transparent"];
            node [shape=box, style="rounded,filled", fillcolor="#F8FAFC", color="#CBD5E1", fontname="Helvetica"];
            edge [color="#64748B", fontname="Helvetica"];

            articles [label="Newswire articles"];
            provider [label="Embedding model\\nOpenAI or Hugging Face"];
            stored [label="Stored article vectors\\n.npy matrix"];
            query [label="User search query"];
            query_vector [label="Query vector\\nStreamlit cached"];
            similarity [label="Cosine similarity\\nquery vs articles"];
            results [label="Ranked articles"];

            articles -> provider -> stored;
            query -> provider -> query_vector;
            stored -> similarity;
            query_vector -> similarity;
            similarity -> results;
        }
        """,
        width="stretch",
    )


st.title("Vector Search")
st.write(
    "Compare semantic search results from the OpenAI and Hugging Face embedding "
    "sets side by side."
)

with st.expander("How vector search works", expanded=True):
    render_process_diagram()
    st.write(
        "The article embeddings are created ahead of time and stored locally. "
        "At search time, the query is embedded with the same provider, cached by "
        "Streamlit for repeated searches, and compared to every article vector "
        "with cosine similarity."
    )

missing_paths = [
    path
    for path in [RAW_SAMPLE_PATH, OPENAI_EMBEDDINGS_PATH, HF_EMBEDDINGS_PATH]
    if not path.exists()
]

if missing_paths:
    st.error("Generate the local data and embeddings before using vector search.")
    for path in missing_paths:
        st.code(str(path.relative_to(ROOT)))
    st.stop()

articles = load_articles(RAW_SAMPLE_PATH)
openai_embeddings = load_embeddings(OPENAI_EMBEDDINGS_PATH)
hf_embeddings = load_embeddings(HF_EMBEDDINGS_PATH)

if len(articles) != len(openai_embeddings) or len(articles) != len(hf_embeddings):
    st.error(
        "The article sample and embedding matrices have different row counts. "
        "Regenerate the data artifacts before searching."
    )
    st.write(
        {
            "articles": len(articles),
            "openai_embeddings": len(openai_embeddings),
            "hf_embeddings": len(hf_embeddings),
        }
    )
    st.stop()

query = st.text_input(
    "Search query",
    placeholder="Example: labor protests over government regulation",
)
k = st.slider("Results per provider", min_value=3, max_value=10, value=5)

artifact_cols = st.columns(3)
artifact_cols[0].metric("Articles", f"{len(articles):,}")
artifact_cols[1].metric("OpenAI dimensions", f"{openai_embeddings.shape[1]:,}")
artifact_cols[2].metric("HF dimensions", f"{hf_embeddings.shape[1]:,}")

clean_query = query.strip()

if not clean_query:
    st.info("Enter a query to compare OpenAI and Hugging Face search results.")
else:
    with st.spinner("Embedding query and searching both vector spaces..."):
        openai_query = cached_query_embedding("openai", clean_query)
        hf_query = cached_query_embedding("hf", clean_query)

    openai_column, hf_column = st.columns(2)

    with openai_column:
        render_results(
            "OpenAI embeddings",
            openai_query,
            openai_embeddings,
            articles,
            k,
        )

    with hf_column:
        render_results(
            "Hugging Face embeddings",
            hf_query,
            hf_embeddings,
            articles,
            k,
        )
