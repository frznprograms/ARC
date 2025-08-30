#!/usr/bin/env python3
"""
Convert labeled review samples (CSV) into fastText .txt files with train/eval split.

Input (CSV with header):
  Columns:
    - "spam", "ad", "irrelevant", "rant", "unsafe"  (0=positive, 1=negative)
    - "text" (string)

Output:
  Directory structure:
    out_dir/train/spam.txt, ad.txt, irrelevant.txt, rant.txt, unsafe.txt
    out_dir/eval/spam.txt, ad.txt, irrelevant.txt, rant.txt, unsafe.txt
"""

import argparse
import csv
import os
import sys
import random
from typing import Iterable, Dict, List, Any, Tuple

CATEGORIES = ["spam", "ad", "irrelevant", "rant", "unsafe"]

def load_samples_csv(path: str) -> List[Dict[str, Any]]:
    """Load samples from a CSV file with required headers."""
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")
        headers_lower = {h.lower().strip(): h for h in reader.fieldnames}

        missing = [col for col in CATEGORIES + ["text"] if col not in headers_lower]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        hmap = {canon: headers_lower[canon] for canon in CATEGORIES + ["text"]}

        for i, row in enumerate(reader, start=2):
            if not any((row[h] or "").strip() for h in row):
                continue

            sample: Dict[str, Any] = {}
            for cat in CATEGORIES:
                raw = (row[hmap[cat]] or "").strip()
                if raw in ("0", "1"):
                    sample[cat] = int(raw)
                else:
                    try:
                        val = int(float(raw))
                        if val not in (0, 1):
                            raise ValueError
                        sample[cat] = val
                    except Exception:
                        raise ValueError(
                            f"Row {i}: column '{hmap[cat]}' must be 0 or 1, got {raw!r}"
                        )
            sample["text"] = (row[hmap["text"]] or "").strip()
            samples.append(sample)
    return samples

def normalize_text(s: str) -> str:
    return " ".join(s.replace("\t", " ").replace("\n", " ").split())

def validate_sample(idx: int, sample: Dict[str, Any]) -> None:
    missing = [k for k in CATEGORIES + ["text"] if k not in sample]
    if missing:
        raise ValueError(f"Sample #{idx} missing keys: {missing}")
    for k in CATEGORIES:
        v = sample[k]
        if v not in (0, 1):
            raise ValueError(f"Sample #{idx} key '{k}' must be 0 or 1, got {v!r}")
    if not isinstance(sample["text"], str):
        raise ValueError(f"Sample #{idx} 'text' must be a string.")

def gen_fasttext_line(label_val: int, text: str) -> str:
    label = "__label__pos" if label_val == 1 else "__label__neg"
    return f"{label} {text}"

def split_samples(samples: List[Dict[str, Any]], train_ratio: float = 0.9) -> Tuple[List, List]:
    random.shuffle(samples)
    cutoff = int(len(samples) * train_ratio)
    return samples[:cutoff], samples[cutoff:]

def write_outputs(samples: Iterable[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    files = {
        cat: open(os.path.join(out_dir, f"{cat}.txt"), "w", encoding="utf-8")
        for cat in CATEGORIES
    }
    try:
        for i, s in enumerate(samples, start=1):
            validate_sample(i, s)
            text = normalize_text(s["text"])
            if not text:
                continue
            for cat in CATEGORIES:
                print(gen_fasttext_line(s[cat], text), file=files[cat])
    finally:
        for f in files.values():
            f.close()

def main():
    ap = argparse.ArgumentParser(description="Convert labeled reviews (CSV) to fastText files with train/eval split.")
    ap.add_argument("input_path", help="Path to CSV file of samples")
    ap.add_argument("-o", "--out-dir", default="fasttext_out", help="Directory to write output files")
    args = ap.parse_args()

    samples = load_samples_csv(args.input_path)
    if not samples:
        print("No samples found; nothing to write.", file=sys.stderr)
        sys.exit(1)

    train_samples, eval_samples = split_samples(samples, train_ratio=0.9)

    write_outputs(train_samples, os.path.join(args.out_dir, "train"))
    write_outputs(eval_samples, os.path.join(args.out_dir, "eval"))

    print(f"Done. Wrote train ({len(train_samples)}) and eval ({len(eval_samples)}) samples into {args.out_dir}/train and {args.out_dir}/eval")

if __name__ == "__main__":
    main()
