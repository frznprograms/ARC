from dataclasses import dataclass
from loguru import logger
import re
import pandas as pd
from typing import Union
from pathlib import Path

from eda.toxic_eda.base_preprocess import BaseProcessor


@dataclass
class TwitterProcessor(BaseProcessor):
    def preprocess(self, save_path: Union[str, Path]):
        logger.info("Starting preprocessing...")

        clean_data = pd.DataFrame()
        clean_data["text"] = self.raw_dataset["tweet"].apply(self._clean_tweet)
        clean_data["unsafe_label"] = 1  # all of them are unsafe
        clean_data = clean_data[["text", "unsafe_label"]]  # type:ignore

        clean_data.drop_duplicates(inplace=True)

        self.save_data(data=clean_data, path=save_path)  # type: ignore
        logger.success("Preprocessed Twitter dataset successfully.")

    def _clean_tweet(self, text: str) -> str:
        if not isinstance(text, str):
            return text

        # remove triple quotes
        text = re.sub(r'"+', "", text)

        # remove @mentions and http links
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"http\S+", "", text)

        # remove anything that starts with & or !
        text = re.sub(r"&+\S*", "", text)
        text = re.sub(r"!+\S*", "", text)

        # remove extra spaces and other random weird inclusions
        text = re.sub(r"RT", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text
