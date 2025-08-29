from dataclasses import dataclass

from src.eda.toxic_eda.base_preprocess import BaseProcessor
from loguru import logger
from typing import Union
from pathlib import Path

import pandas as pd


@dataclass
class JigsawProcessor(BaseProcessor):

    # since each dataset has its own unique columns and properties,
    # it is likely more efficient to handle each dataset separately
    def preprocess(self, save_path: Union[str, Path]):
        logger.info("Starting preprocessing...")
        self._add_unsafe_label(self.raw_dataset)

        self.raw_dataset.drop_duplicates(subset=["id", "comment_text"], inplace=True)
        cleaned_dataset = self.raw_dataset[["comment_text", "unsafe_label"]]
        cleaned_dataset.columns = ["text", "unsafe_label"]

        self.save_data(data=cleaned_dataset, path=save_path)  # type: ignore
        logger.success("Preprocessed Jigsaw dataset successfully.")

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
