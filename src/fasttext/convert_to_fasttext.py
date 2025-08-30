#!/usr/bin/env python3
"""
Convert labeled review samples into fastText .txt files.

Input:
- A JSON file that is either:
  (a) a list of dicts, or
  (b) JSONL (one dict per line).
Each dict has keys:
  - "spam", "ad", "irrelevant", "rant", "unsafe" (0=positive, 1=negative)
  - "text" (string)

Output:
- Five files in the output directory:
  spam.txt, ad.txt, irrelevant.txt, rant.txt, unsafe.txt
  Each line is: "__label__pos <space> <text>" or "__label__neg <space> <text>"
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List

CATEGORIES = ["spam", "ad", "irrelevant", "rant", "unsafe"]


def load_samples(path: str) -> List[Dict[str, Any]]:
    """Load samples from a JSON array or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON array
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Top-level JSON must be a list of dicts.")
            return data
        else:
            # JSONL
            samples = []
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL on line {i}: {e}") from e
                samples.append(obj)
            return samples


def normalize_text(s: str) -> str:
    """Light normalization for fastText: collapse whitespace & strip."""
    # Replace newlines/tabs with spaces, collapse runs of whitespace.
    return " ".join(s.replace("\t", " ").replace("\n", " ").split())


def validate_sample(idx: int, sample: Dict[str, Any]) -> None:
    """Basic validation to catch missing keys / bad types early."""
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
    """
    Map 0/1 to __label__pos / __label__neg per user spec (0=positive, 1=negative),
    and produce a single fastText line.
    """
    label = "__label__pos" if label_val == 0 else "__label__neg"
    return f"{label} {text}"


def write_outputs(samples: Iterable[Dict[str, Any]], out_dir: str) -> None:
    """Write one file per category with fastText lines."""
    os.makedirs(out_dir, exist_ok=True)

    # Open all output files once.
    files = {
        cat: open(os.path.join(out_dir, f"{cat}.txt"), "w", encoding="utf-8")
        for cat in CATEGORIES
    }
    try:
        for i, s in enumerate(samples, start=1):
            validate_sample(i, s)
            text = normalize_text(s["text"])
            if not text:
                # Skip entirely empty text after normalization
                continue

            for cat in CATEGORIES:
                label_val = s[cat]  # 0=pos, 1=neg (as specified)
                line = gen_fasttext_line(label_val, text)
                print(line, file=files[cat])
    finally:
        for f in files.values():
            f.close()


def main():
    ap = argparse.ArgumentParser(
        description="Convert labeled reviews to fastText files."
    )
    ap.add_argument("input_path", help="Path to JSON array or JSONL file of samples")
    ap.add_argument(
        "-o",
        "--out-dir",
        default="fasttext_out",
        help="Directory to write output files (default: fasttext_out)",
    )
    args = ap.parse_args()

    samples = load_samples(args.input_path)
    if not samples:
        print("No samples found; nothing to write.", file=sys.stderr)
        sys.exit(1)

    write_outputs(samples, args.out_dir)
    print(
        f"Done. Wrote: {', '.join(cat + '.txt' for cat in CATEGORIES)} in {args.out_dir}"
    )


if __name__ == "__main__":
    main()
