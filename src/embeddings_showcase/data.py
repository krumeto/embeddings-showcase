"""Dataset loading and local artifact helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


TEXT_COLUMN = "article"
CATEGORY_COLUMNS = [
    "antitrust",
    "civil_rights",
    "crime",
    "govt_regulation",
    "labor_movement",
    "politics",
    "protests",
]
RARE_LABELS = ["antitrust", "civil_rights", "protests"]
OTHER_LABELS = ["crime", "govt_regulation", "labor_movement", "politics"]


def load_newswire_dataframe(
    data_files: Iterable[str] = ("1973_data_clean.json",),
    split: str = "train",
) -> pd.DataFrame:
    """Load the labelled newswire dataset into a compact dataframe."""
    from datasets import load_dataset

    dataset = load_dataset(
        "dell-research-harvard/newswire",
        data_files=list(data_files),
        split=split,
    )
    return dataset.select_columns([TEXT_COLUMN, *CATEGORY_COLUMNS]).to_pandas()


def sample_labelled_articles(
    df: pd.DataFrame,
    target_n: int = 5_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a balanced-ish labelled sample without extra dependencies."""
    labelled = df[df[CATEGORY_COLUMNS].sum(axis=1) > 0].copy()

    rare_mask = labelled[RARE_LABELS].sum(axis=1) > 0
    sample = labelled[rare_mask].copy()
    remaining_n = max(target_n - len(sample), 0)

    remaining = labelled[~labelled.index.isin(sample.index)].copy()
    other_mask = remaining[OTHER_LABELS].sum(axis=1) > 0
    remaining = remaining[other_mask]

    if remaining_n == 0 or remaining.empty:
        return sample.drop_duplicates().head(target_n)

    remaining["sampling_label"] = remaining[OTHER_LABELS].idxmax(axis=1)
    label_proportions = remaining["sampling_label"].value_counts(normalize=True)
    n_per_label = (label_proportions * remaining_n).round().astype(int)

    parts = []
    for label, n in n_per_label.items():
        label_df = remaining[remaining["sampling_label"] == label]
        n = min(int(n), len(label_df))
        if n:
            parts.append(label_df.sample(n=n, random_state=random_state))

    if parts:
        sample = pd.concat([sample, *parts], axis=0)

    return (
        sample.drop_duplicates()
        .drop(columns=["sampling_label"], errors="ignore")
        .head(target_n)
    )


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
