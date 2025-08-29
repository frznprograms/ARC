from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger


@dataclass
class BaseProcessor(ABC):
    file_path: Union[str, Path]
    raw_dataset: pd.DataFrame = field(default_factory=pd.DataFrame)
    debug_mode: bool = False

    @logger.catch(message="Unable to read dataset.", reraise=True)
    def read_data(self, path: Union[str, Path], **kwargs) -> pd.DataFrame:
        data = pd.read_csv(path, kwargs)  # type: ignore
        return data

    @logger.catch(
        message="Unable to save dataset. Please ensure the path is valid and that the dataset can \
                be converted to the appropriate file type.",
        reraise=True,
    )
    def save_data(self, data: pd.DataFrame, path: Union[str, Path]) -> None:
        logger.info("Saving dataset to path...")
        data.to_csv(path, index=False)
        logger.success(f"Dataset saved to {path}.")

    def _get_value_counts(self):
        for col in self.raw_dataset.columns:
            try:
                print(self.raw_dataset[col].value_counts())
                print("-" * 50)
            except Exception:
                logger.warning(
                    "There was an issue printing the value counts of the `{col}` column which is of \
                    type {self.raw_dataset[col].dtype}. Please ensure it is of a numeric type that can \
                    be analysed using `pandas.Series.value_counts()`. Alternatively, specify the names \
                    of the columns you would like to view in the function call."
                )
