from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from embeddings_showcase.data import CATEGORY_COLUMNS
from embeddings_showcase.embeddings import embed_with_huggingface, embed_with_openai


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


st.set_page_config(
    page_title="Classification",
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
def train_classifier(provider: str, dimension_label: str) -> dict[str, object]:
    articles = load_articles(RAW_SAMPLE_PATH)
    embeddings = provider_embeddings(provider, dimension_label)
    labels = articles[CATEGORY_COLUMNS].astype(int).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.25,
        random_state=42,
    )

    model = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=42,
        )
    )
    model.fit(x_train, y_train)

    return {
        "model": model,
        "x_test": x_test,
        "y_test": y_test,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "dimensions": x_train.shape[1],
        "provider": provider,
    }


@st.cache_data(show_spinner=False)
def cached_text_embedding(provider: str, text: str) -> np.ndarray:
    if provider == "openai":
        return embed_with_openai([text])[0]
    if provider == "hf":
        return embed_with_huggingface([text], timeout=60)[0]
    raise ValueError(f"Unknown provider: {provider}")


def metrics_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = [
        {
            "metric": "micro precision",
            "value": precision_score(y_true, y_pred, average="micro", zero_division=0),
        },
        {
            "metric": "micro recall",
            "value": recall_score(y_true, y_pred, average="micro", zero_division=0),
        },
        {
            "metric": "micro f1",
            "value": f1_score(y_true, y_pred, average="micro", zero_division=0),
        },
        {
            "metric": "macro f1",
            "value": f1_score(y_true, y_pred, average="macro", zero_division=0),
        },
        {
            "metric": "hamming loss",
            "value": hamming_loss(y_true, y_pred),
        },
        {
            "metric": "exact match accuracy",
            "value": accuracy_score(y_true, y_pred),
        },
    ]
    return pd.DataFrame(rows)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "micro precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming loss": hamming_loss(y_true, y_pred),
        "exact match accuracy": accuracy_score(y_true, y_pred),
    }


def comparison_table(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        for metric, value in metrics_dict(result["y_test"], result["y_pred"]).items():
            rows.append(
                {
                    "provider": name,
                    "metric": metric,
                    "value": value,
                }
            )
    return (
        pd.DataFrame(rows)
        .pivot(index="metric", columns="provider", values="value")
        .reset_index()
    )


def per_label_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=[label.replace("_", " ") for label in CATEGORY_COLUMNS],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in [label.replace("_", " ") for label in CATEGORY_COLUMNS]:
        rows.append(
            {
                "label": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
        )
    return pd.DataFrame(rows)


def probabilities_table(probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    for label, probability in zip(CATEGORY_COLUMNS, probabilities, strict=True):
        rows.append(
            {
                "label": label.replace("_", " "),
                "probability": float(probability),
                "predicted": bool(probability >= threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("probability", ascending=False)


st.title("Classification")
st.write(
    "Train a simple multilabel logistic regression classifier on top of the "
    "stored OpenAI and Hugging Face embeddings."
)

missing_paths = [
    path
    for path in [RAW_SAMPLE_PATH, OPENAI_EMBEDDINGS_PATH, HF_EMBEDDINGS_PATH]
    if not path.exists()
]
if missing_paths:
    st.error("Generate the raw sample and both embedding files before using classification.")
    for path in missing_paths:
        st.code(str(path.relative_to(ROOT)))
    st.stop()

dimension_label = st.select_slider(
    "OpenAI embedding dimensions",
    options=list(DIMENSION_OPTIONS.keys()),
    value="384",
    help=(
        "OpenAI text-embedding-3 models are Matryoshka-style embeddings, so the "
        "first dimensions can be sliced down for faster experiments."
    ),
)
threshold = st.slider(
    "Prediction threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.5,
    step=0.05,
)

with st.spinner("Training logistic regression models..."):
    openai_trained = train_classifier("openai", dimension_label)
    hf_trained = train_classifier("hf", dimension_label)

results = {}
for name, trained in {
    "OpenAI": openai_trained,
    "Hugging Face": hf_trained,
}.items():
    probabilities = trained["model"].predict_proba(trained["x_test"])
    y_pred = (probabilities >= threshold).astype(int)
    results[name] = {
        **trained,
        "probabilities": probabilities,
        "y_pred": y_pred,
    }

openai_trained = results["OpenAI"]
hf_trained = results["Hugging Face"]

metric_cols = st.columns(4)
metric_cols[0].metric("Training rows", f"{openai_trained['train_rows']:,}")
metric_cols[1].metric("Test rows", f"{openai_trained['test_rows']:,}")
metric_cols[2].metric(
    "Dimensions",
    f"OpenAI {openai_trained['dimensions']:,} / HF {hf_trained['dimensions']:,}",
)
metric_cols[3].metric("Labels", f"{len(CATEGORY_COLUMNS):,}")

st.subheader("Metric Comparison")
st.dataframe(
    comparison_table(results).style.format(
        {
            "OpenAI": "{:.3f}",
            "Hugging Face": "{:.3f}",
        }
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Per-Provider Details")
openai_metrics_col, hf_metrics_col = st.columns(2)

with openai_metrics_col:
    st.markdown("### OpenAI")
    st.caption(f"{openai_trained['dimensions']:,} dimensions")
    st.dataframe(
        metrics_table(openai_trained["y_test"], openai_trained["y_pred"]).style.format(
            {"value": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        per_label_report(openai_trained["y_test"], openai_trained["y_pred"]).style.format(
            {
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with hf_metrics_col:
    st.markdown("### Hugging Face")
    st.caption(f"{hf_trained['dimensions']:,} dimensions")
    st.dataframe(
        metrics_table(hf_trained["y_test"], hf_trained["y_pred"]).style.format(
            {"value": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        per_label_report(hf_trained["y_test"], hf_trained["y_pred"]).style.format(
            {
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.subheader("Try Your Own Text")
example_text = st.text_area(
    "Article text",
    height=180,
    placeholder=(
        "Paste or write a short news-style paragraph. The app will embed it with "
        "both providers and predict labels side by side."
    ),
)

clean_text = example_text.strip()
if not clean_text:
    st.info("Enter example text to see predicted labels.")
else:
    with st.spinner("Embedding example text and predicting labels..."):
        openai_vector = cached_text_embedding("openai", clean_text)
        if DIMENSION_OPTIONS[dimension_label] is not None:
            openai_vector = openai_vector[: DIMENSION_OPTIONS[dimension_label]]
        hf_vector = cached_text_embedding("hf", clean_text)

        openai_example_probabilities = openai_trained["model"].predict_proba(
            openai_vector.reshape(1, -1)
        )[0]
        hf_example_probabilities = hf_trained["model"].predict_proba(
            hf_vector.reshape(1, -1)
        )[0]

    prediction_cols = st.columns(2)
    for column, title, example_probabilities in [
        (prediction_cols[0], "OpenAI", openai_example_probabilities),
        (prediction_cols[1], "Hugging Face", hf_example_probabilities),
    ]:
        with column:
            st.markdown(f"### {title}")
            prediction_df = probabilities_table(example_probabilities, threshold)
            predicted_labels = prediction_df[prediction_df["predicted"]]["label"].tolist()

            if predicted_labels:
                st.success(f"Predicted labels: {', '.join(predicted_labels)}")
            else:
                st.warning("No labels crossed the current threshold.")

            st.dataframe(
                prediction_df.style.format({"probability": "{:.3f}"}),
                hide_index=True,
                width="stretch",
            )
