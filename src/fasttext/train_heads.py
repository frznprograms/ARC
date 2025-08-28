#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import fasttext

CATEGORIES = ["spam", "ad", "irrelevant", "rant", "unsafe"]

def train_one(input_path: str, lr: float, epoch: int, dim: int, minn: int, maxn: int,
              wordNgrams: int, autotune: bool, autotune_duration: int):
    """
    Train a single fastText supervised model on input_path.
    If autotune=True and a file named <name>_valid.txt exists next to input_path,
    use it as a validation set for autotuning within autotune_duration seconds.
    """
    if autotune:
        # Look for a sibling validation file like spam_valid.txt
        p = Path(input_path)
        valid_path = p.with_name(p.stem + "_valid.txt")
        if valid_path.exists():
            print(f"[autotune] Using validation file: {valid_path}")
            model = fasttext.train_supervised(
                input=str(input_path),
                autotuneValidationFile=str(valid_path),
                autotuneDuration=autotune_duration
            )
            return model
        else:
            print(f"[autotune] No validation file for {p.name}; falling back to fixed params.")

    # Fixed hyperparams (fast, reasonable defaults)
    model = fasttext.train_supervised(
        input=str(input_path),
        lr=lr,
        epoch=epoch,
        dim=dim,
        minn=minn,
        maxn=maxn,
        wordNgrams=wordNgrams,
        loss="softmax"
    )
    return model

def main():
    ap = argparse.ArgumentParser(description="Train one fastText model per category/head.")
    ap.add_argument("--data-dir", required=True, help="Folder with spam.txt, ad.txt, ...")
    ap.add_argument("--out-dir", required=True, help="Folder to save .bin models")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--minn", type=int, default=2, help="min char ngram")
    ap.add_argument("--maxn", type=int, default=5, help="max char ngram")
    ap.add_argument("--wordNgrams", type=int, default=2)
    ap.add_argument("--autotune", action="store_true", help="Use autotune if *_valid.txt exists")
    ap.add_argument("--autotune-duration", type=int, default=60, help="seconds for autotune")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for cat in CATEGORIES:
        in_file = os.path.join(args.data_dir, f"{cat}.txt")
        if not os.path.exists(in_file):
            raise FileNotFoundError(f"Missing training file: {in_file}")

        print(f"==> Training head: {cat}")
        model = train_one(
            in_file, args.lr, args.epoch, args.dim, args.minn, args.maxn,
            args.wordNgrams, args.autotune, args.autotune_duration
        )
        out_path = os.path.join(args.out_dir, f"{cat}.bin")
        model.save_model(out_path)
        print(f"[saved] {out_path}")

    print("All heads trained & saved.")

if __name__ == "__main__":
    main()
