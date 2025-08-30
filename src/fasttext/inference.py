#!/usr/bin/env python3
"""
Inference script for multi-head fastText classifier (hard OR gate)
with per-head enable/disable flags.

Usage:
  python inference.py --model-dir models --text "your review"
  python inference.py --model-dir models --text "..." --spam false --rant true
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

# Adjust if needed
from fasttext_classifier import FasttextClassifier


def parse_thresholds(s: str) -> Dict[str, float]:
    """Parse a string like 'spam:0.7,ad:0.6' into a dict."""
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid per-head threshold '{part}'. Expected format cat:float"
            )
        cat, val = part.split(":", 1)
        out[cat.strip()] = float(val.strip())
    return out


def discover_categories(model_dir: Path):
    """Find all *.bin files and use their stems as category names."""
    cats = []
    for p in sorted(model_dir.glob("*.bin")):
        if p.is_file():
            cats.append(p.stem)
    if not cats:
        raise FileNotFoundError(f"No .bin models found in '{model_dir}'")
    return cats


def main():
    # First parse known args (to get model-dir and text)
    ap = argparse.ArgumentParser(
        description="Infer good/bad with multi-head fastText OR-gate."
    )
    ap.add_argument(
        "--model-dir", required=True, help="Directory with <category>.bin heads."
    )
    ap.add_argument("--text", required=True, help="Input string to classify.")
    ap.add_argument(
        "--default-threshold",
        type=float,
        default=None,
        help="Default probability threshold for all heads.",
    )
    ap.add_argument(
        "--per-head-thresholds",
        type=parse_thresholds,
        default=None,
        help='Optional per-head thresholds, e.g. "spam:0.7,ad:0.6".',
    )
    args, unknown = ap.parse_known_args()

    model_dir = Path(args.model_dir)
    categories = discover_categories(model_dir)

    # Build dynamic parser for enable/disable flags
    ap2 = argparse.ArgumentParser()
    for cat in categories:
        ap2.add_argument(
            f"--{cat}",
            type=str,
            choices=["true", "false"],
            help=f"Enable/disable {cat} head (default: true).",
        )
    flags = ap2.parse_args(unknown)

    # Convert to dict of {cat: bool}
    enabled: Dict[str, bool] = {}
    for cat in categories:
        val = getattr(flags, cat)
        enabled[cat] = (val.lower() != "false") if val is not None else True

    # Filter categories based on enabled flags
    active_categories = [c for c in categories if enabled[c]]
    if not active_categories:
        raise RuntimeError("All heads disabled; nothing to run inference on.")

    clf = FasttextClassifier(categories=active_categories, model_dir=model_dir)

    label, fired = clf.predict_or_gate(
        args.text,
        default_threshold=args.default_threshold,
        threshold_per_head=args.per_head_thresholds,
        return_triggering_heads=True,
    )

    result = {
        "label": label,
        "fired_heads": fired,
        "disabled_heads": [c for c in categories if not enabled[c]],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
