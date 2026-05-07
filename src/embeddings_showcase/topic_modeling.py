"""Topic modeling helpers built around Turftopic Topeax."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import Topeax
from turftopic.encoders import ExternalEncoder

from embeddings_showcase.data import CATEGORY_COLUMNS


TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,}")


@dataclass(frozen=True)
class TopeaxFitResult:
    model: Topeax
    document_topic_matrix: np.ndarray
    topic_ids: np.ndarray
    topic_strengths: np.ndarray
    topic_names: list[str]
    topic_words: list[list[str]]
    topic_weights: np.ndarray
    reduced_embeddings: np.ndarray


class CorpusTermEncoder(ExternalEncoder):
    """Embed vocabulary terms by averaging document embeddings containing them.

    Topeax can receive precomputed document embeddings, but it still needs an
    encoder for candidate topic terms. This keeps term embeddings local and in
    the same coordinate system as the stored article embeddings.
    """

    def __init__(self, documents: Iterable[str], embeddings: np.ndarray):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.fallback = self.embeddings.mean(axis=0)
        self.token_sets = [
            set(TOKEN_PATTERN.findall(str(document).lower()))
            for document in documents
        ]

    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        vectors = []
        for sentence in sentences:
            terms = TOKEN_PATTERN.findall(str(sentence).lower())
            matches = [
                index
                for index, tokens in enumerate(self.token_sets)
                if terms and all(term in tokens for term in terms)
            ]
            if matches:
                vectors.append(self.embeddings[matches].mean(axis=0))
            else:
                vectors.append(self.fallback)
        return np.asarray(vectors, dtype=np.float32)


def active_labels(row: pd.Series) -> str:
    labels = [label.replace("_", " ") for label in CATEGORY_COLUMNS if row.get(label, 0)]
    return ", ".join(labels) if labels else "unlabelled"


def article_preview(text: str, max_chars: int = 500) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def fit_topeax_topics(
    documents: list[str],
    embeddings: np.ndarray,
    max_features: int,
    min_df: int,
    max_df: float,
    perplexity: int,
    random_state: int = 42,
) -> TopeaxFitResult:
    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
    )
    model = Topeax(
        encoder=CorpusTermEncoder(documents, embeddings),
        vectorizer=vectorizer,
        perplexity=perplexity,
        random_state=random_state,
    )
    document_topic_matrix = model.fit_transform(documents, embeddings=embeddings)
    topic_ids = document_topic_matrix.argmax(axis=1)
    topic_strengths = document_topic_matrix.max(axis=1)
    topic_words = model.get_top_words(10)

    return TopeaxFitResult(
        model=model,
        document_topic_matrix=document_topic_matrix,
        topic_ids=topic_ids,
        topic_strengths=topic_strengths,
        topic_names=list(model.topic_names),
        topic_words=topic_words,
        topic_weights=np.asarray(model.weights_),
        reduced_embeddings=np.asarray(model.reduced_embeddings),
    )


def build_topic_frame(
    articles: pd.DataFrame,
    fit: TopeaxFitResult,
) -> pd.DataFrame:
    frame = articles.reset_index(drop=True).copy()
    frame.insert(0, "row_id", np.arange(len(frame)))
    frame["known_labels"] = frame.apply(active_labels, axis=1)
    frame["topic_id"] = fit.topic_ids.astype(int)
    frame["topic_name"] = [fit.topic_names[topic_id] for topic_id in fit.topic_ids]
    frame["topic_strength"] = fit.topic_strengths
    frame["x"] = fit.reduced_embeddings[:, 0]
    frame["y"] = fit.reduced_embeddings[:, 1]
    frame["preview"] = frame["article"].map(article_preview)
    return frame


def topic_summary_frame(topic_frame: pd.DataFrame, fit: TopeaxFitResult) -> pd.DataFrame:
    rows = []
    for topic_id, topic_name in enumerate(fit.topic_names):
        topic_docs = topic_frame[topic_frame["topic_id"] == topic_id]
        rows.append(
            {
                "topic_id": topic_id,
                "topic": topic_name,
                "weight": float(fit.topic_weights[topic_id]),
                "articles": len(topic_docs),
                "avg_strength": float(topic_docs["topic_strength"].mean()),
                "top_words": ", ".join(fit.topic_words[topic_id][:8]),
                "dominant_label": dominant_label(topic_docs["known_labels"]),
            }
        )
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


def dominant_label(labels: pd.Series) -> str:
    exploded = []
    for value in labels:
        exploded.extend(label.strip() for label in str(value).split(",") if label.strip())
    if not exploded:
        return "unlabelled"
    return str(pd.Series(exploded).value_counts().index[0])


def label_distribution(topic_docs: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for labels in topic_docs["known_labels"]:
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
    topic_frame: pd.DataFrame,
    topic_id: int,
    limit: int = 8,
) -> pd.DataFrame:
    return (
        topic_frame[topic_frame["topic_id"] == topic_id]
        .sort_values("topic_strength", ascending=False)
        .head(limit)
        [["row_id", "known_labels", "topic_strength", "preview"]]
    )
