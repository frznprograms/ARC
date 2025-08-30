#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import fasttext


class FasttextHead:
    """
    Single fastText head for one category.

    Assumes binary labels in the training file (e.g., __label__pos / __label__neg).
    `positive_label` defines which label counts as "positive" for this head.
    """

    def __init__(
        self,
        label: str,
        model_path: Optional[Union[str, Path]] = None,
        positive_label: str = "__label__pos",
    ):
        self.label: str = label
        self.positive_label: str = positive_label
        self.model: Optional[fasttext.FastText._FastText] = None
        if model_path is not None:
            self.load(model_path)

    # ----------------------- IO -----------------------
    def load(self, model_path: Union[str, Path]) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found for head '{self.label}': {model_path}"
            )
        self.model = fasttext.load_model(str(model_path))

    def save(self, out_path: Union[str, Path]) -> None:
        if self.model is None:
            raise RuntimeError(
                f"Head '{self.label}' has no trained/loaded model to save."
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out_path))

    # --------------------- Training -------------------
    def train(
        self,
        input_path: Union[str, Path],
        *,
        lr: float = 0.5,
        epoch: int = 10,
        dim: int = 100,
        minn: int = 2,
        maxn: int = 5,
        wordNgrams: int = 2,
        autotune: bool = False,
        autotune_duration: int = 60,
        valid_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Train on `input_path`. If autotune is True and a valid file is present
        (or <stem>_valid.txt), use fastText autotune.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Training file not found for head '{self.label}': {input_path}"
            )

        if autotune:
            if valid_path is None:
                valid_path = input_path.with_name(input_path.stem + "_valid.txt")
            valid_path = Path(valid_path)
            if valid_path.exists():
                print(f"[{self.label}] autotune with validation: {valid_path}")
                self.model = fasttext.train_supervised(
                    input=str(input_path),
                    autotuneValidationFile=str(valid_path),
                    autotuneDuration=autotune_duration,
                )
                return
            else:
                print(
                    f"[{self.label}] no validation file at {valid_path}; using fixed params."
                )

        print(f"[{self.label}] training with fixed hyperparams.")
        self.model = fasttext.train_supervised(
            input=str(input_path),
            lr=lr,
            epoch=epoch,
            dim=dim,
            minn=minn,
            maxn=maxn,
            wordNgrams=wordNgrams,
            loss="softmax",
        )

    # -------------------- Inference -------------------
    def predict_raw(
        self,
        texts: Union[str, Iterable[str]],
        *,
        k: int = 1,
    ) -> Union[Tuple[List[str], List[float]], List[Tuple[List[str], List[float]]]]:
        """
        Raw fastText predict wrapper.
        """
        if self.model is None:
            raise RuntimeError(f"Head '{self.label}' has no model loaded/trained.")
        if isinstance(texts, str):
            labels, probs = self.model.predict(texts, k=k)
            return list(labels), list(probs)
        out = []
        for t in texts:
            labels, probs = self.model.predict(t, k=k)
            out.append((list(labels), list(probs)))
        return out

    def is_positive(self, text: str, *, threshold: Optional[float] = None) -> bool:
        """
        Returns True iff the top-1 predicted label == positive_label and its prob >= threshold (if set).
        (Assumes binary classification per head.)
        """
        labels, probs = self.predict_raw(text, k=1)  # top-1
        if not labels:
            return False
        if labels[0] != self.positive_label:
            return False
        return (threshold is None) or (probs[0] >= threshold)  # type: ignore

    def is_positive_batch(
        self, texts: Iterable[str], *, threshold: Optional[float] = None
    ) -> List[bool]:
        """
        Batch version of is_positive.
        """
        raw = self.predict_raw(list(texts), k=1)  # list of (labels, probs)
        out: List[bool] = []
        for labels, probs in raw:
            if (
                labels
                and labels[0] == self.positive_label  # type: ignore
                and ((threshold is None) or (probs[0] >= threshold))  # type: ignore
            ):
                out.append(True)
            else:
                out.append(False)
        return out
