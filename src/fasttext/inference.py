#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Tuple
import fasttext

CATEGORIES = ["spam", "ad", "irrelevant", "rant", "unsafe"]
POS_LABEL = "__label__pos"

def load_models(models_dir: str) -> Dict[str, fasttext.FastText._FastText]:
    models = {}
    for cat in CATEGORIES:
        path = os.path.join(models_dir, f"{cat}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}")
        models[cat] = fasttext.load_model(path)
    return models

def predict_heads(models: Dict[str, fasttext.FastText._FastText], text: str,
                  threshold: float = 0.5) -> Dict[str, Tuple[str, float, bool]]:
    """
    Returns per-head results:
      {category: (pred_label, prob, fired_bool)}
    fired_bool is True if pred_label == __label__pos and prob >= threshold.
    """
    results: Dict[str, Tuple[str, float, bool]] = {}
    for cat, model in models.items():
        labels, probs = model.predict(text, k=1)
        label = labels[0]
        prob = float(probs[0])
        fired = (label == POS_LABEL) and (prob >= threshold)
        results[cat] = (label, prob, fired)
    return results

def aggregate_or(results: Dict[str, Tuple[str, float, bool]]) -> bool:
    """True if any head fired (BAD), else False (GOOD)."""
    return any(fired for (_, _, fired) in results.values())

def main():
    ap = argparse.ArgumentParser(description="OR-gate inference over 5 fastText heads.")
    ap.add_argument("--models-dir", required=True, help="Folder containing *.bin models")
    ap.add_argument("--text", required=True, help="Review text to classify")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Min prob to treat __label__pos as a 'yes' for a head")
    args = ap.parse_args()

    models = load_models(args.models_dir)
    results = predict_heads(models, args.text, threshold=args.threshold)
    is_bad = aggregate_or(results)

    print("Per-head predictions:")
    for cat, (label, prob, fired) in results.items():
        print(f"  {cat:11s} -> {label:12s} p={prob:.3f} fired={fired}")

    print(f"\nFINAL: {'BAD (rejected)' if is_bad else 'GOOD (accepted)'}")

if __name__ == "__main__":
    main()
