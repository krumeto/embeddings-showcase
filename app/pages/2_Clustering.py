from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from embeddings_showcase.clustering import (
    EvocFitResult,
    build_article_cluster_frame,
    build_treemap_frame,
    cluster_display_name,
    display_layer_columns,
    fit_evoc_clusters,
    label_distribution,
    layer_summary,
    representative_articles,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_SAMPLE_PATH = ROOT / "data" / "raw" / "newswire_5000_sample.parquet"
OPENAI_EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "openai_embeddings.npy"
HF_EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "hf_embeddings.npy"
DIMENSION_OPTIONS = {
    "384": 384,
    "512": 512,
    "1024": 1024,
    "full": None,
}
PROVIDER_OPTIONS = {
    "OpenAI": "openai",
    "Hugging Face": "hf",
}


st.set_page_config(
    page_title="Clustering",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_articles(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def slice_dimensions(vectors: np.ndarray, dimension_label: str) -> np.ndarray:
    dimensions = DIMENSION_OPTIONS[dimension_label]
    if dimensions is None:
        return vectors
    return vectors[:, :dimensions]


def provider_embeddings(provider: str, dimension_label: str) -> np.ndarray:
    if provider == "openai":
        embeddings = load_embeddings(OPENAI_EMBEDDINGS_PATH)
        return slice_dimensions(embeddings, dimension_label)
    if provider == "hf":
        return load_embeddings(HF_EMBEDDINGS_PATH)
    raise ValueError(f"Unknown provider: {provider}")


@st.cache_resource(show_spinner=False)
def cached_evoc_fit(
    provider: str,
    dimension_label: str,
    base_min_cluster_size: int,
    n_neighbors: int,
    max_layers: int,
    noise_level: float,
) -> EvocFitResult:
    vectors = provider_embeddings(provider, dimension_label)
    return fit_evoc_clusters(
        vectors=vectors,
        base_min_cluster_size=base_min_cluster_size,
        n_neighbors=n_neighbors,
        max_layers=max_layers,
        noise_level=noise_level,
        random_state=42,
    )


def run_key(
    provider: str,
    dimension_label: str,
    base_min_cluster_size: int,
    n_neighbors: int,
    max_layers: int,
    noise_level: float,
) -> tuple[str, str, int, int, int, float]:
    return (
        provider,
        dimension_label,
        base_min_cluster_size,
        n_neighbors,
        max_layers,
        round(noise_level, 2),
    )


def format_ratio(value: float) -> str:
    return f"{value:.1%}"


st.title("Clustering")
st.write(
    "EVōC finds topic structure at multiple granularities: broad news themes at "
    "the top, more specific subtopics as you drill down."
)

with st.expander("How this clustering view works", expanded=True):
    st.write(
        "The page fits EVōC to the stored article embeddings, then reverses the "
        "learned cluster layers into a broad-to-fine hierarchy. The treemap uses "
        "rectangle size for article count, and the detail pane shows the strongest "
        "representative articles for one selected cluster."
    )
    st.write(
        "Articles assigned to `-1` are shown as noise, meaning EVōC did not place "
        "them confidently inside a cluster at that layer."
    )

missing_paths = [
    path
    for path in [RAW_SAMPLE_PATH, OPENAI_EMBEDDINGS_PATH, HF_EMBEDDINGS_PATH]
    if not path.exists()
]

if missing_paths:
    st.error("Generate the local data and embeddings before using clustering.")
    for path in missing_paths:
        st.code(str(path.relative_to(ROOT)))
    st.stop()

articles = load_articles(RAW_SAMPLE_PATH)

with st.sidebar:
    st.header("EVōC Settings")
    provider_label = st.selectbox("Embedding provider", list(PROVIDER_OPTIONS.keys()))
    provider = PROVIDER_OPTIONS[provider_label]
    dimension_label = st.select_slider(
        "OpenAI dimensions",
        options=list(DIMENSION_OPTIONS.keys()),
        value="384",
        disabled=provider != "openai",
        help=(
            "OpenAI text-embedding-3 vectors can be sliced to their first "
            "dimensions for faster experiments. Hugging Face uses its stored "
            "384-dimensional vectors."
        ),
    )
    if provider == "hf":
        dimension_label = "full"

    base_min_cluster_size = st.slider(
        "Minimum cluster size",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
    )
    n_neighbors = st.slider(
        "Neighbors",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )
    max_layers = st.slider(
        "Maximum layers",
        min_value=1,
        max_value=5,
        value=3,
    )
    noise_level = st.slider(
        "Noise level",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    current_key = run_key(
        provider,
        dimension_label,
        base_min_cluster_size,
        n_neighbors,
        max_layers,
        noise_level,
    )
    run_requested = st.button("Run clustering", type="primary")

if run_requested:
    st.session_state["clustering_run_key"] = current_key

if "clustering_run_key" not in st.session_state:
    artifact_cols = st.columns(3)
    artifact_cols[0].metric("Articles", f"{len(articles):,}")
    artifact_cols[1].metric(
        "OpenAI embeddings",
        "Ready" if OPENAI_EMBEDDINGS_PATH.exists() else "Missing",
    )
    artifact_cols[2].metric(
        "HF embeddings",
        "Ready" if HF_EMBEDDINGS_PATH.exists() else "Missing",
    )
    st.info("Choose EVōC settings in the sidebar, then run clustering.")
    st.stop()

selected_key = st.session_state["clustering_run_key"]
if selected_key != current_key:
    st.warning("Settings changed. Run clustering again to update the hierarchy.")

(
    selected_provider,
    selected_dimension_label,
    selected_base_min_cluster_size,
    selected_n_neighbors,
    selected_max_layers,
    selected_noise_level,
) = selected_key

with st.spinner("Fitting EVōC hierarchy... first runs include Numba compilation."):
    fit = cached_evoc_fit(
        selected_provider,
        selected_dimension_label,
        selected_base_min_cluster_size,
        selected_n_neighbors,
        selected_max_layers,
        selected_noise_level,
    )

article_cluster_df = build_article_cluster_frame(articles, fit)
tree_df, path_columns = build_treemap_frame(article_cluster_df)
layer_columns = display_layer_columns(article_cluster_df)
finest_layer = "layer_0"
finest_summary = layer_summary(article_cluster_df, finest_layer)

metric_cols = st.columns(5)
metric_cols[0].metric("Articles", f"{len(article_cluster_df):,}")
metric_cols[1].metric("Hierarchy layers", f"{len(fit.cluster_layers):,}")
metric_cols[2].metric("Fine clusters", f"{finest_summary['clusters']:,}")
metric_cols[3].metric("Fine noise", f"{finest_summary['noise_count']:,}")
metric_cols[4].metric("Noise share", format_ratio(finest_summary["noise_ratio"]))

fig = px.treemap(
    tree_df,
    path=path_columns,
    values="count",
    color="dominant_label",
    hover_data={
        "count": True,
        "avg_strength": ":.3f",
        "dominant_label": True,
        "label_purity": ":.1%",
    },
)
fig.update_traces(root_color="#F1F5F9", textinfo="label")
fig.update_layout(
    margin=dict(t=12, l=8, r=8, b=8),
    height=680,
)

st.subheader("Broad-To-Fine Topic Map")
st.caption(
    "Short structural labels stay on the chart; hover carries count, dominant "
    "known label, label purity, and average EVōC membership strength."
)
st.plotly_chart(fig, width="stretch")

st.subheader("Cluster Details")
detail_cols = st.columns([1, 1, 2])

with detail_cols[0]:
    selected_layer_column = st.selectbox(
        "Layer",
        options=layer_columns,
        format_func=lambda column: f"Layer {column.split('_')[1]}",
    )

available_cluster_ids = sorted(
    int(cluster_id) for cluster_id in article_cluster_df[selected_layer_column].unique()
)
non_noise_ids = [cluster_id for cluster_id in available_cluster_ids if cluster_id != -1]
default_cluster_id = non_noise_ids[0] if non_noise_ids else available_cluster_ids[0]

with detail_cols[1]:
    selected_cluster_id = st.selectbox(
        "Cluster",
        options=available_cluster_ids,
        index=available_cluster_ids.index(default_cluster_id),
        format_func=lambda cluster_id: cluster_display_name(
            int(selected_layer_column.split("_")[1]),
            cluster_id,
        ),
    )

selected_cluster_df = article_cluster_df[
    article_cluster_df[selected_layer_column] == selected_cluster_id
]
selected_summary = layer_summary(article_cluster_df, selected_layer_column)

with detail_cols[2]:
    st.metric("Clusters in selected layer", f"{selected_summary['clusters']:,}")
    st.metric("Selected cluster size", f"{len(selected_cluster_df):,}")

label_df = label_distribution(selected_cluster_df)
representatives = representative_articles(
    article_cluster_df,
    selected_layer_column,
    selected_cluster_id,
)

label_col, representative_col = st.columns([1, 2])

with label_col:
    st.markdown("### Known Label Mix")
    if label_df.empty:
        st.info("No known labels for this cluster.")
    else:
        st.dataframe(label_df, hide_index=True, width="stretch")

with representative_col:
    st.markdown("### Representative Articles")
    st.dataframe(
        representatives.style.format({"membership_strength": "{:.3f}"}),
        hide_index=True,
        width="stretch",
    )
