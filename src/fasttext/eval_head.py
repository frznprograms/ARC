#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from src.fasttext.fasttext_head import FasttextHead
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate one FasttextHead on a labeled test file."
    )
    ap.add_argument(
        "--category",
        required=True,
        help="Category name of the head (for logging only).",
    )
    ap.add_argument(
        "--model", required=True, help="Path to trained .bin fastText model."
    )
    ap.add_argument(
        "--test-file", required=True, help="Test set (fastText supervised format)."
    )
    ap.add_argument(
        "--positive-label",
        default="__label__pos",
        help="Label in test file to treat as positive (default: __label__pos).",
    )
    ap.add_argument(
        "--neg-label",
        default=None,
        help="Optional explicit negative label. If omitted, anything not positive is treated as negative.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional probability threshold for positive (default: use top-1).",
    )
    return ap.parse_args()


def load_test_data(path: Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split on ANY whitespace (space or tab), once
            parts = line.split(None, 1)
            if not parts:
                continue
            label = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            labels.append(label)
            texts.append(text)
    return texts, labels


def main():
    args = parse_args()
    test_path = Path(args.test_file)

    head = FasttextHead(
        label=args.category,
        model_path=args.model,
        positive_label=args.positive_label,
    )

    texts, labels_true = load_test_data(test_path)

    # Map arbitrary string labels -> binary 1 (pos) / 0 (neg)
    pos = args.positive_label
    neg = args.neg_label  # may be None

    # True labels (binary)
    y_true_bin = [1 if y == pos else 0 for y in labels_true]

    # Predicted labels (binary)
    y_pred_bin = [
        1 if head.is_positive(t, threshold=args.threshold) else 0 for t in texts
    ]

    # Confusion matrix with fixed binary labels [1, 0] avoids string mismatches
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[1, 0])
    precision = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    # Nice printout
    print(f"== Evaluation for head: {args.category} ==")
    print(f"Positive label in file: {pos}")
    if neg:
        print(f"Negative label in file: {neg} (used only for display)")
    print("Confusion matrix (rows: true, cols: pred) for classes [POS(1), NEG(0)]:")
    print(cm)
    print(f"Precision (POS): {precision:.4f}")
    print(f"Recall    (POS): {recall:.4f}")

    # Optional: warn if the expected positive label never appears in y_true
    if sum(y_true_bin) == 0:
        print(
            "[warn] No positive examples found in y_true. Precision is computed on predicted positives only; "
            "recall will be 0. Check --positive-label or your test file."
        )
    if sum(1 - y for y in y_true_bin) == 0:
        print(
            "[warn] No negative examples found in y_true. Metrics may be degenerate; "
            "consider a more balanced test set."
        )


if __name__ == "__main__":
    main()
