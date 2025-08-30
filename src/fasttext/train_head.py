#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fasttext_head import FasttextHead


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train a single FasttextHead for one category."
    )
    # Required I/O
    ap.add_argument(
        "--category", required=True, help="Category name for this head (e.g., spam)."
    )
    ap.add_argument("--train-file", required=True, help="Path to training text file.")
    ap.add_argument("--out", required=True, help="Output .bin model path.")

    # Optional preload / positive label
    ap.add_argument(
        "--preload", default=None, help="Existing .bin to warm-start this head."
    )
    ap.add_argument(
        "--positive-label",
        default="__label__pos",
        help="Which label counts as positive (default: __label__pos).",
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
        help="Use fastText autotune if a valid file is provided or discovered.",
    )
    ap.add_argument(
        "--autotune-duration", type=int, default=60, help="seconds for autotune"
    )
    ap.add_argument(
        "--valid-file",
        default=None,
        help="Optional validation file for autotune. "
        "If omitted, will try <train_file_stem>_valid.txt when --autotune is set.",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    train_path = Path(args.train_file)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    head = FasttextHead(
        label=args.category,
        model_path=args.preload,
        positive_label=args.positive_label,
    )

    head.train(
        input_path=train_path,
        lr=args.lr,
        epoch=args.epoch,
        dim=args.dim,
        minn=args.minn,
        maxn=args.maxn,
        wordNgrams=args.wordNgrams,
        autotune=args.autotune,
        autotune_duration=args.autotune_duration,
        valid_path=args.valid_file,  # None -> head will auto-discover <stem>_valid.txt
    )

    head.save(out_path)
    print(f"[saved] {out_path.resolve()}")


if __name__ == "__main__":
    main()
