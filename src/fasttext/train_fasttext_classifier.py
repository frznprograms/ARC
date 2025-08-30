#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from fasttext_head import FasttextHead

DEFAULT_CATEGORIES = ["spam", "ad", "irrelevant", "rant", "unsafe"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train multiple FasttextHead models sequentially (one head at a time)."
    )

    # I/O
    ap.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing <category>.txt (and optional <category>_valid.txt).",
    )
    ap.add_argument(
        "--out-dir", required=True, help="Directory to save <category>.bin models."
    )
    ap.add_argument(
        "--categories",
        nargs="*",
        default=DEFAULT_CATEGORIES,
        help=f"List of categories. Default: {DEFAULT_CATEGORIES}",
    )
    ap.add_argument(
        "--preload-dir",
        default=None,
        help="Optional directory to warm-start from existing <category>.bin if present.",
    )
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip categories whose training file is missing instead of raising an error.",
    )

    # Labels
    ap.add_argument(
        "--positive-label",
        default="__label__pos",
        help="Positive class label used by each head. Default: __label__pos",
    )

    # Hyperparams
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--minn", type=int, default=2, help="min char ngram")
    ap.add_argument("--maxn", type=int, default=5, help="max char ngram")
    ap.add_argument("--wordNgrams", type=int, default=2)

    # Autotune
    ap.add_argument(
        "--autotune",
        action="store_true",
        help="Use fastText autotune if <category>_valid.txt exists (or is provided via --valid-suffix).",
    )
    ap.add_argument(
        "--autotune-duration", type=int, default=60, help="Seconds to spend autotuning."
    )
    ap.add_argument(
        "--valid-suffix",
        default="_valid.txt",
        help="Validation filename suffix (e.g., '_dev.txt'). Default: _valid.txt",
    )

    return ap.parse_args()


def train_heads_sequential(
    categories: Iterable[str],
    data_dir: Path,
    out_dir: Path,
    *,
    preload_dir: Optional[Path],
    positive_label: str,
    lr: float,
    epoch: int,
    dim: int,
    minn: int,
    maxn: int,
    wordNgrams: int,
    autotune: bool,
    autotune_duration: int,
    valid_suffix: str,
    skip_missing: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in categories:
        train_file = data_dir / f"{cat}.txt"
        if not train_file.exists():
            msg = f"[warn] Missing training file for category '{cat}': {train_file}"
            if skip_missing:
                print(msg + " — skipping.")
                continue
            else:
                raise FileNotFoundError(msg)

        preload_path = None
        if preload_dir is not None:
            candidate = preload_dir / f"{cat}.bin"
            if candidate.exists():
                preload_path = candidate

        print(f"\n==> Training head: {cat}")
        if preload_path:
            print(f"[{cat}] warm-start from: {preload_path}")

        head = FasttextHead(
            label=cat, model_path=preload_path, positive_label=positive_label
        )

        valid_path = None
        if autotune:
            # e.g., spam.txt -> spam_valid.txt (or custom suffix)
            valid_path = train_file.with_name(train_file.stem + valid_suffix)

        head.train(
            input_path=train_file,
            lr=lr,
            epoch=epoch,
            dim=dim,
            minn=minn,
            maxn=maxn,
            wordNgrams=wordNgrams,
            autotune=autotune,
            autotune_duration=autotune_duration,
            valid_path=valid_path,  # head will ignore if not exists
        )

        out_path = out_dir / f"{cat}.bin"
        head.save(out_path)
        print(f"[saved] {out_path.resolve()}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    preload_dir = Path(args.preload_dir) if args.preload_dir else None

    train_heads_sequential(
        categories=args.categories,
        data_dir=data_dir,
        out_dir=out_dir,
        preload_dir=preload_dir,
        positive_label=args.positive_label,
        lr=args.lr,
        epoch=args.epoch,
        dim=args.dim,
        minn=args.minn,
        maxn=args.maxn,
        wordNgrams=args.wordNgrams,
        autotune=args.autotune,
        autotune_duration=args.autotune_duration,
        valid_suffix=args.valid_suffix,
        skip_missing=args.skip_missing,
    )

    print("\nAll requested heads processed.")


if __name__ == "__main__":
    main()
