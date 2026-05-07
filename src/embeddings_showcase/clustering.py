"""Small clustering helpers for embedding demos."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def cluster_kmeans(
    vectors: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    return model.fit_predict(vectors)
