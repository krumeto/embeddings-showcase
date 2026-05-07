"""EVōC clustering helpers for the Streamlit clustering demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from evoc import EVoC

from embeddings_showcase.data import CATEGORY_COLUMNS


@dataclass(frozen=True)
class EvocFitResult:
    clusterer: EVoC
    labels: np.ndarray
    membership_strengths: np.ndarray
    cluster_layers: list[np.ndarray]
    membership_strength_layers: list[np.ndarray]


def fit_evoc_clusters(
    vectors: np.ndarray,
    base_min_cluster_size: int,
    n_neighbors: int,
    max_layers: int,
    noise_level: float,
    random_state: int = 42,
) -> EvocFitResult:
    """Fit EVōC and return the fitted hierarchy in a small typed container."""
    clusterer = EVoC(
        base_min_cluster_size=base_min_cluster_size,
        n_neighbors=n_neighbors,
        max_layers=max_layers,
        noise_level=noise_level,
        random_state=random_state,
    )
    labels = clusterer.fit_predict(vectors)
    cluster_layers = [np.asarray(layer) for layer in clusterer.cluster_layers_]
    if not cluster_layers:
        cluster_layers = [np.asarray(labels)]

    membership_strength_layers = [
        np.asarray(layer) for layer in clusterer.membership_strength_layers_
    ]
    if not membership_strength_layers:
        membership_strength_layers = [np.asarray(clusterer.membership_strengths_)]

    return EvocFitResult(
        clusterer=clusterer,
        labels=np.asarray(labels),
        membership_strengths=np.asarray(clusterer.membership_strengths_),
        cluster_layers=cluster_layers,
        membership_strength_layers=membership_strength_layers,
    )


def active_labels(row: pd.Series) -> str:
    labels = [label.replace("_", " ") for label in CATEGORY_COLUMNS if row.get(label, 0)]
    return ", ".join(labels) if labels else "unlabelled"


def article_preview(text: str, max_chars: int = 450) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def cluster_display_name(layer_index: int, cluster_id: int) -> str:
    if int(cluster_id) == -1:
        return f"Layer {layer_index} / Noise"
    return f"Layer {layer_index} / Cluster {int(cluster_id)}"


def build_article_cluster_frame(
    articles: pd.DataFrame,
    fit: EvocFitResult,
) -> pd.DataFrame:
    """Combine article metadata with EVōC layer assignments."""
    frame = articles.reset_index(drop=True).copy()
    frame.insert(0, "row_id", np.arange(len(frame)))
    frame["known_labels"] = frame.apply(active_labels, axis=1)
    frame["membership_strength"] = fit.membership_strengths

    for layer_index, layer in enumerate(fit.cluster_layers):
        frame[f"layer_{layer_index}"] = layer.astype(int)
        if layer_index < len(fit.membership_strength_layers):
            frame[f"layer_{layer_index}_strength"] = fit.membership_strength_layers[
                layer_index
            ]

    return frame


def layer_columns(article_cluster_df: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in article_cluster_df.columns
        if column.startswith("layer_")
        and column.removeprefix("layer_").isdigit()
    ]
    return sorted(columns, key=lambda column: int(column.split("_")[1]))


def display_layer_columns(article_cluster_df: pd.DataFrame) -> list[str]:
    """Return layer columns from broad/coarse to fine for hierarchy rendering."""
    return list(reversed(layer_columns(article_cluster_df)))


def dominant_label(labels: pd.Series) -> str:
    exploded = []
    for value in labels:
        exploded.extend(label.strip() for label in str(value).split(",") if label.strip())

    if not exploded:
        return "unlabelled"

    counts = pd.Series(exploded).value_counts()
    return str(counts.index[0])


def label_purity(labels: pd.Series) -> float:
    exploded = []
    for value in labels:
        exploded.extend(label.strip() for label in str(value).split(",") if label.strip())

    if not exploded:
        return 0.0

    counts = pd.Series(exploded).value_counts()
    return float(counts.iloc[0] / counts.sum())


def build_treemap_frame(article_cluster_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate article assignments into Plotly Express treemap paths."""
    display_columns = display_layer_columns(article_cluster_df)
    tree_df = article_cluster_df[["row_id", "known_labels", "membership_strength"]].copy()
    tree_df["All articles"] = "All articles"

    path_columns = ["All articles"]
    for column in display_columns:
        layer_index = int(column.split("_")[1])
        display_column = f"Layer {layer_index}"
        tree_df[display_column] = article_cluster_df[column].map(
            lambda cluster_id, index=layer_index: cluster_display_name(
                index,
                int(cluster_id),
            )
        )
        path_columns.append(display_column)

    grouped = (
        tree_df.groupby(path_columns, dropna=False)
        .agg(
            count=("row_id", "size"),
            avg_strength=("membership_strength", "mean"),
            dominant_label=("known_labels", dominant_label),
            label_purity=("known_labels", label_purity),
        )
        .reset_index()
    )
    return grouped, path_columns


def layer_summary(article_cluster_df: pd.DataFrame, layer_column: str) -> dict[str, Any]:
    labels = article_cluster_df[layer_column]
    noise_count = int((labels == -1).sum())
    cluster_count = int(labels[labels != -1].nunique())
    return {
        "clusters": cluster_count,
        "noise_count": noise_count,
        "noise_ratio": noise_count / len(article_cluster_df),
    }


def label_distribution(article_cluster_df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for labels in article_cluster_df["known_labels"]:
        for label in str(labels).split(","):
            clean = label.strip()
            if clean:
                counts[clean] = counts.get(clean, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["label", "articles"])

    return (
        pd.DataFrame(
            [{"label": label, "articles": count} for label, count in counts.items()]
        )
        .sort_values("articles", ascending=False)
        .reset_index(drop=True)
    )


def representative_articles(
    article_cluster_df: pd.DataFrame,
    layer_column: str,
    cluster_id: int,
    limit: int = 8,
) -> pd.DataFrame:
    strength_column = f"{layer_column}_strength"
    sort_column = (
        strength_column
        if strength_column in article_cluster_df.columns
        else "membership_strength"
    )
    cluster_df = article_cluster_df[article_cluster_df[layer_column] == cluster_id].copy()
    cluster_df["preview"] = cluster_df["article"].map(article_preview)

    return (
        cluster_df.sort_values(sort_column, ascending=False)
        .head(limit)
        [["row_id", "known_labels", sort_column, "preview"]]
        .rename(columns={sort_column: "membership_strength"})
    )
