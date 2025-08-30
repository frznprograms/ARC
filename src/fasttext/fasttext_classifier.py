#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from src.fasttext.fasttext_head import FasttextHead


class FasttextClassifier:
    """
    Multi-head classifier with a hard OR gate:
      If ANY head is positive -> returns "good"
      Else -> returns "bad"

    You can pass per-head positive labels via `positive_label_map`,
    otherwise defaults to "__label__pos".
    """

    def __init__(
        self,
        categories: Iterable[str],
        model_dir: Optional[Union[str, Path]] = None,
        positive_label_map: Optional[Dict[str, str]] = None,
    ):
        self.categories: List[str] = list(categories)
        self.heads: Dict[str, FasttextHead] = {}

        model_dir_path = Path(model_dir) if model_dir is not None else None

        for cat in self.categories:
            pos_label = "__label__pos"
            if positive_label_map and cat in positive_label_map:
                pos_label = positive_label_map[cat]

            model_path = None
            if model_dir_path is not None:
                mp = model_dir_path / f"{cat}.bin"
                if mp.exists():
                    model_path = mp

            self.heads[cat] = FasttextHead(
                label=cat, model_path=model_path, positive_label=pos_label
            )

    # --------------------- Training ---------------------
    def train_all(
        self,
        data_dir: Union[str, Path],
        *,
        lr: float = 0.5,
        epoch: int = 10,
        dim: int = 100,
        minn: int = 2,
        maxn: int = 5,
        wordNgrams: int = 2,
        autotune: bool = False,
        autotune_duration: int = 60,
    ) -> None:
        data_dir = Path(data_dir)
        for cat, head in self.heads.items():
            train_file = data_dir / f"{cat}.txt"
            if not train_file.exists():
                raise FileNotFoundError(
                    f"Missing training file for category '{cat}': {train_file}"
                )
            print(f"==> Training head: {cat}")
            head.train(
                train_file,
                lr=lr,
                epoch=epoch,
                dim=dim,
                minn=minn,
                maxn=maxn,
                wordNgrams=wordNgrams,
                autotune=autotune,
                autotune_duration=autotune_duration,
                valid_path=None,
            )

    # ----------------------- IO -------------------------
    def save_all(self, out_dir: Union[str, Path]) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cat, head in self.heads.items():
            out_path = out_dir / f"{cat}.bin"
            head.save(out_path)
            print(f"[saved] {out_path}")

    # -------------------- OR-gate Inference -------------
    def predict_or_gate(
        self,
        texts: Union[str, Iterable[str]],
        *,
        default_threshold: Optional[float] = None,
        threshold_per_head: Optional[Dict[str, float]] = None,
        return_triggering_heads: bool = False,
    ):
        """
        If any head is positive -> "bad", else "good".
        Optionally return which heads triggered for transparency.

        Returns:
          - str if `texts` is a single string
          - List[str] if `texts` is an iterable of strings
          - If `return_triggering_heads` is True:
              - Tuple[str, List[str]] or List[Tuple[str, List[str]]]
                where second element lists heads that fired.
        """

        def th(cat: str) -> Optional[float]:
            if threshold_per_head and cat in threshold_per_head:
                return threshold_per_head[cat]
            return default_threshold

        def decide_one(text: str):
            fired = []
            for cat, head in self.heads.items():
                if head.is_positive(text, threshold=th(cat)):
                    fired.append(cat)
            label = "bad" if len(fired) > 0 else "good"
            return (label, fired) if return_triggering_heads else label

        if isinstance(texts, str):
            return decide_one(texts)
        else:
            return [decide_one(t) for t in texts]
