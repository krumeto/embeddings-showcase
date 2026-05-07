from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from embeddings_showcase.topic_modeling import (
    TopeaxFitResult,
    build_topic_frame,
    fit_topeax_topics,
    label_distribution,
    representative_articles,
    topic_summary_frame,
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
    page_title="Topic Modelling",
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


def sample_rows(total_rows: int, sample_size: int, random_state: int = 42) -> np.ndarray:
    if sample_size >= total_rows:
        return np.arange(total_rows)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(total_rows, size=sample_size, replace=False))


@st.cache_resource(show_spinner=False)
def cached_topeax_fit(
    provider: str,
    dimension_label: str,
    sample_size: int,
    max_chars: int,
    max_features: int,
    min_df: int,
    max_df: float,
    perplexity: int,
) -> dict[str, object]:
    articles = load_articles(RAW_SAMPLE_PATH)
    embeddings = provider_embeddings(provider, dimension_label)
    indices = sample_rows(len(articles), sample_size)
    sampled_articles = articles.iloc[indices].reset_index(drop=True)
    sampled_embeddings = embeddings[indices]
    documents = [
        " ".join(str(article).split())[:max_chars]
        for article in sampled_articles["article"]
    ]
    fit = fit_topeax_topics(
        documents=documents,
        embeddings=sampled_embeddings,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        perplexity=perplexity,
        random_state=42,
    )
    topic_frame = build_topic_frame(sampled_articles, fit)
    summary = topic_summary_frame(topic_frame, fit)
    return {
        "fit": fit,
        "topic_frame": topic_frame,
        "summary": summary,
        "sampled_rows": indices,
        "dimensions": sampled_embeddings.shape[1],
    }


def run_key(
    provider: str,
    dimension_label: str,
    sample_size: int,
    max_chars: int,
    max_features: int,
    min_df: int,
    max_df: float,
    perplexity: int,
) -> tuple[str, str, int, int, int, int, float, int]:
    return (
        provider,
        dimension_label,
        sample_size,
        max_chars,
        max_features,
        min_df,
        round(max_df, 2),
        perplexity,
    )


st.title("Topic Modelling")
st.write(
    "Topeax discovers topics as density peaks in a two-dimensional embedding map, "
    "then estimates topic words by combining lexical evidence with the embedding "
    "space around each topic."
)

with st.expander("How this page follows the Topeax example", expanded=True):
    st.write(
        "The linked Turftopic example fits `Topeax`, calls `plot_steps(...)`, and "
        "prints the highest-ranking terms for each topic. This page uses the same "
        "model and visualization pattern, but passes the already-generated local "
        "article embeddings into `fit_transform(..., embeddings=...)`."
    )
    st.write(
        "Because the app already has document embeddings, vocabulary terms are "
        "embedded locally by averaging the embeddings of articles where each term "
        "appears. That keeps the topic-word scoring in the same vector space "
        "without downloading another encoder model."
    )

missing_paths = [
    path
    for path in [RAW_SAMPLE_PATH, OPENAI_EMBEDDINGS_PATH, HF_EMBEDDINGS_PATH]
    if not path.exists()
]
if missing_paths:
    st.error("Generate the local data and embeddings before using topic modelling.")
    for path in missing_paths:
        st.code(str(path.relative_to(ROOT)))
    st.stop()

articles = load_articles(RAW_SAMPLE_PATH)

with st.sidebar:
    st.header("Topeax Settings")
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

    sample_size = st.slider(
        "Articles to model",
        min_value=250,
        max_value=min(2_000, len(articles)),
        value=min(750, len(articles)),
        step=250,
    )
    max_chars = st.slider(
        "Characters per article",
        min_value=500,
        max_value=4_000,
        value=1_500,
        step=500,
    )
    perplexity = st.slider(
        "t-SNE perplexity",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
    )
    max_features = st.slider(
        "Vocabulary size",
        min_value=500,
        max_value=5_000,
        value=2_000,
        step=500,
    )
    min_df = st.slider(
        "Minimum term documents",
        min_value=2,
        max_value=20,
        value=5,
    )
    max_df = st.slider(
        "Maximum term document share",
        min_value=0.30,
        max_value=0.95,
        value=0.75,
        step=0.05,
    )

    current_key = run_key(
        provider,
        dimension_label,
        sample_size,
        max_chars,
        max_features,
        min_df,
        max_df,
        perplexity,
    )
    run_requested = st.button("Run topic model", type="primary")

if run_requested:
    st.session_state["topic_model_run_key"] = current_key

if "topic_model_run_key" not in st.session_state:
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
    st.info("Choose Topeax settings in the sidebar, then run the topic model.")
    st.stop()

selected_key = st.session_state["topic_model_run_key"]
if selected_key != current_key:
    st.warning("Settings changed. Run the topic model again to update the view.")

with st.spinner("Fitting Topeax topic model..."):
    result = cached_topeax_fit(*selected_key)

fit: TopeaxFitResult = result["fit"]
topic_frame: pd.DataFrame = result["topic_frame"]
summary: pd.DataFrame = result["summary"]

metric_cols = st.columns(5)
metric_cols[0].metric("Modelled articles", f"{len(topic_frame):,}")
metric_cols[1].metric("Discovered topics", f"{len(fit.topic_names):,}")
metric_cols[2].metric("Embedding dimensions", f"{result['dimensions']:,}")
metric_cols[3].metric("Vocabulary terms", f"{fit.model.components_.shape[1]:,}")
metric_cols[4].metric("Largest topic share", f"{fit.topic_weights.max():.1%}")

st.subheader("Topeax Algorithm Steps")
st.caption(
    "This mirrors Turftopic's `plot_steps(...)`: t-SNE document map, density "
    "peaks, Gaussian mixture components, and component probabilities."
)
steps_fig = fit.model.plot_steps(
    hover_text=[
        f"row {row.row_id} · {row.known_labels}<br>{row.preview[:220]}"
        for row in topic_frame.itertuples()
    ]
)
steps_fig.update_layout(height=760)
st.plotly_chart(steps_fig, width="stretch")

st.subheader("Topic Map")
scatter_fig = px.scatter(
    topic_frame,
    x="x",
    y="y",
    color="topic_name",
    hover_data={
        "row_id": True,
        "known_labels": True,
        "topic_strength": ":.3f",
        "preview": True,
        "x": False,
        "y": False,
    },
    opacity=0.75,
)
scatter_fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#1F2937")))
scatter_fig.update_layout(
    height=620,
    margin=dict(t=16, l=8, r=8, b=8),
    legend_title_text="Topic",
)
st.plotly_chart(scatter_fig, width="stretch")

st.subheader("Topics")
st.dataframe(
    summary.style.format(
        {
            "weight": "{:.1%}",
            "avg_strength": "{:.3f}",
        }
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Topic Details")
topic_options = summary["topic_id"].astype(int).tolist()
selected_topic_id = st.selectbox(
    "Topic",
    topic_options,
    format_func=lambda topic_id: fit.topic_names[topic_id],
)
topic_docs = topic_frame[topic_frame["topic_id"] == selected_topic_id]

detail_cols = st.columns(3)
detail_cols[0].metric("Articles", f"{len(topic_docs):,}")
detail_cols[1].metric(
    "Topic weight",
    f"{fit.topic_weights[selected_topic_id]:.1%}",
)
detail_cols[2].metric(
    "Average strength",
    f"{topic_docs['topic_strength'].mean():.3f}",
)

word_col, label_col, article_col = st.columns([1, 1, 2])
with word_col:
    st.markdown("### Top Words")
    st.write(", ".join(fit.topic_words[selected_topic_id]))

with label_col:
    st.markdown("### Known Label Mix")
    label_df = label_distribution(topic_docs)
    if label_df.empty:
        st.info("No known labels for this topic.")
    else:
        st.dataframe(label_df, hide_index=True, width="stretch")

with article_col:
    st.markdown("### Representative Articles")
    st.dataframe(
        representative_articles(topic_frame, selected_topic_id).style.format(
            {"topic_strength": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )

st.subheader("All Articles In Topic")
explorer_cols = st.columns([1, 1, 2])
with explorer_cols[0]:
    sort_option = st.selectbox(
        "Sort articles",
        ["Highest topic strength", "Lowest topic strength", "Row order"],
    )
with explorer_cols[1]:
    label_options = sorted(
        {
            label.strip()
            for labels in topic_docs["known_labels"]
            for label in str(labels).split(",")
            if label.strip()
        }
    )
    selected_labels = st.multiselect(
        "Filter labels",
        label_options,
        placeholder="All labels",
    )
with explorer_cols[2]:
    search_text = st.text_input(
        "Search within topic",
        placeholder="Find words inside article previews",
    )

article_explorer_df = topic_docs.copy()
if selected_labels:
    article_explorer_df = article_explorer_df[
        article_explorer_df["known_labels"].map(
            lambda labels: any(label in str(labels) for label in selected_labels)
        )
    ]

clean_search = search_text.strip().lower()
if clean_search:
    article_explorer_df = article_explorer_df[
        article_explorer_df["preview"].str.lower().str.contains(clean_search, regex=False)
    ]

if sort_option == "Highest topic strength":
    article_explorer_df = article_explorer_df.sort_values(
        "topic_strength",
        ascending=False,
    )
elif sort_option == "Lowest topic strength":
    article_explorer_df = article_explorer_df.sort_values("topic_strength")
else:
    article_explorer_df = article_explorer_df.sort_values("row_id")

st.caption(
    f"Showing {len(article_explorer_df):,} of {len(topic_docs):,} articles assigned "
    f"to {fit.topic_names[selected_topic_id]}."
)
st.dataframe(
    article_explorer_df[
        ["row_id", "known_labels", "topic_strength", "preview"]
    ].style.format({"topic_strength": "{:.3f}"}),
    hide_index=True,
    width="stretch",
    height=520,
)
