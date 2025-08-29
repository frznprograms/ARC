from dataclasses import dataclass

from eda.toxic_eda.base_preprocess import BaseProcessor
from loguru import logger
from typing import Union
from pathlib import Path

import pandas as pd


@dataclass
class JigsawProcessor(BaseProcessor):

    def preprocess(self, save_path: Union[str, Path]):
        # since we already know that this dataset has its own unique columns, it would be faster to hardcode
        # the preprocessing
        self.raw_dataset.drop_duplicates(subset=["id", "comment_text"], inplace=True)
        self.cleaned_dataset = self.raw_dataset[["comment_text", "unsafe_label"]]
        self.cleaned_dataset.columns = ["text", "unsafe_label"]

        self.save_data(data=self.cleaned_dataset, path=save_path)  # type: ignore

    def _add_unsafe_label(self, df: pd.DataFrame) -> None:
        label_cols = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
        df["unsafe_label"] = df[label_cols].any(axis=1).astype(int)  # type: ignore
        logger.success("Added the appropriate safe/unsafe binary label to dataset.")
