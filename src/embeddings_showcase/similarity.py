"""Simple NumPy vector search utilities."""

from __future__ import annotations

import numpy as np


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, a_min=1e-12, a_max=None)


def cosine_similarity(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    matrix = np.asarray(vectors, dtype=np.float32)
    return normalize_rows(query) @ normalize_rows(matrix).T


def top_k_search(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    k: int = 10,
) -> list[tuple[int, float]]:
    scores = cosine_similarity(query_vector, vectors).ravel()
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(index), float(scores[index])) for index in top_indices]
