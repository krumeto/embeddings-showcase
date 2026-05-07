"""Prepare a labelled local newswire sample for the showcase."""

from __future__ import annotations

import argparse
from pathlib import Path

from embeddings_showcase.data import (
    CATEGORY_COLUMNS,
    load_newswire_dataframe,
    sample_labelled_articles,
    write_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=["1973_data_clean.json"],
        help="Newswire dataset files to load from Hugging Face.",
    )
    parser.add_argument("--target-n", type=int, default=5_000)
    parser.add_argument("--output", default="data/raw/newswire_5000_sample.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_newswire_dataframe(data_files=args.data_files)
    sample = sample_labelled_articles(df, target_n=args.target_n)
    output_path = write_parquet(sample, Path(args.output))

    print(f"Wrote {len(sample):,} rows to {output_path}")
    print(sample[CATEGORY_COLUMNS].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
