"""Embedding provider helpers for OpenAI and Hugging Face."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from huggingface_hub import InferenceClient
from openai import OpenAI

from embeddings_showcase.config import settings


def embed_with_openai(
    texts: Sequence[str],
    model: str | None = None,
) -> np.ndarray:
    """Embed texts with OpenAI and return a 2D NumPy array."""
    if not settings.openai_key:
        raise RuntimeError("OPENAI_KEY is required for OpenAI embeddings.")

    client = OpenAI(api_key=settings.openai_key)
    response = client.embeddings.create(
        model=model or settings.default_openai_model,
        input=list(texts),
    )
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype=np.float32)


def embed_with_huggingface(
    texts: Sequence[str],
    model: str | None = None,
    truncate: bool = True,
    timeout: float | None = 120,
) -> np.ndarray:
    """Embed texts with Hugging Face Inference API and return a 2D NumPy array."""
    if not settings.hf_key:
        raise RuntimeError("HF_KEY is required for Hugging Face embeddings.")

    client = InferenceClient(
        provider="hf-inference",
        api_key=settings.hf_key,
        timeout=timeout,
    )
    vectors = client.feature_extraction(
        list(texts),
        model=model or settings.default_hf_model,
        truncate=truncate,
    )
    return np.asarray(vectors, dtype=np.float32)
