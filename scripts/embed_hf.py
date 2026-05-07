"""Generate Hugging Face embeddings for a prepared parquet dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from embeddings_showcase.data import TEXT_COLUMN, read_parquet
from embeddings_showcase.embeddings import embed_with_huggingface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/newswire_5000_sample.parquet")
    parser.add_argument("--output", default="data/embeddings/hf_embeddings.npy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--save-each-batch",
        action="store_true",
        help="Persist partial progress after every batch.",
    )
    return parser.parse_args()


def batches(items: Sequence[str], batch_size: int) -> list[Sequence[str]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def main() -> None:
    args = parse_args()
    df = read_parquet(args.input)
    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    if args.limit:
        texts = texts[: args.limit]
    if args.max_chars:
        texts = [text[: args.max_chars] for text in texts]

    parts = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for index, batch in enumerate(batches(texts, args.batch_size), start=1):
        vectors = embed_with_huggingface(
            batch,
            model=args.model,
            timeout=args.timeout,
        )
        parts.append(vectors)
        print(f"Embedded Hugging Face batch {index:,} ({len(batch):,} rows)", flush=True)
        if args.save_each_batch:
            np.save(output_path, np.vstack(parts))

    vectors = np.vstack(parts)
    np.save(output_path, vectors)
    print(f"Wrote {vectors.shape} embeddings to {output_path}")


if __name__ == "__main__":
    main()
