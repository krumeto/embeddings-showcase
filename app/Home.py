from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RAW_SAMPLE_PATH = ROOT / "data" / "raw" / "newswire_5000_sample.parquet"
OPENAI_EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "openai_embeddings.npy"
HF_EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "hf_embeddings.npy"


st.set_page_config(
    page_title="Embeddings Showcase",
    page_icon="",
    layout="wide",
)


@st.cache_data
def load_sample_summary(path: Path) -> tuple[int, dict[str, int]] | None:
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    label_columns = [
        "antitrust",
        "civil_rights",
        "crime",
        "govt_regulation",
        "labor_movement",
        "politics",
        "protests",
    ]
    available_labels = [column for column in label_columns if column in df.columns]
    counts = df[available_labels].sum().sort_values(ascending=False).astype(int)
    return len(df), counts.to_dict()


st.sidebar.success("Choose a section above.")

st.title("Embeddings Showcase")
st.write(
    "Explore how dense embeddings can turn a small news archive into a searchable, "
    "clusterable, and classifiable semantic dataset."
)

summary = load_sample_summary(RAW_SAMPLE_PATH)

metric_cols = st.columns(3)
metric_cols[0].metric("Raw sample", "Ready" if RAW_SAMPLE_PATH.exists() else "Missing")
metric_cols[1].metric(
    "OpenAI embeddings",
    "Ready" if OPENAI_EMBEDDINGS_PATH.exists() else "Missing",
)
metric_cols[2].metric(
    "HF embeddings",
    "Ready" if HF_EMBEDDINGS_PATH.exists() else "Missing",
)

if summary:
    rows, label_counts = summary
    st.subheader("Dataset Snapshot")
    st.write(f"The local sample contains **{rows:,}** labelled newswire articles.")
    st.dataframe(
        pd.DataFrame(
            {
                "label": label_counts.keys(),
                "articles": label_counts.values(),
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info(
        "Generate the local data first with `uv run python scripts/prepare_dataset.py`."
    )

st.subheader("Start Exploring")

pages = [
    (
        "Vector search",
        "Find articles by semantic similarity instead of exact keyword overlap.",
        "pages/1_Vector_Search.py",
    ),
    (
        "Clustering",
        "Group nearby article embeddings and inspect emerging themes.",
        "pages/2_Clustering.py",
    ),
    (
        "Topic modelling",
        "Use embeddings to build topic-oriented views of the archive.",
        "pages/3_Topic_Modelling.py",
    ),
    (
        "Classification",
        "Compare embedding-based features against the labelled article categories.",
        "pages/4_Classification.py",
    ),
]

for column, (title, description, target) in zip(st.columns(4), pages, strict=True):
    with column:
        st.markdown(f"### {title}")
        st.write(description)
        st.page_link(target, label=f"Open {title}")
