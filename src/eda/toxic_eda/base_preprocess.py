from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger


@dataclass
class BaseProcessor(ABC):
    file_path: Optional[Union[str, Path]]
    raw_dataset: pd.DataFrame = field(default_factory=pd.DataFrame)
    debug_mode: bool = False

    # TODO: add some stuff for debug mode

    def __post_init__(self):
        if self.file_path is not None:
            # will be disabled for Toxigen dataset, loaded from HF
            self.raw_dataset = self.read_data(self.file_path)

    @abstractmethod
    def preprocess(self, save_path: Union[str, Path]):
        pass

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

    def _get_value_counts(
        self, cols: Optional[list[str]] = None
    ) -> dict[str, pd.Series]:
        if cols is None:
            cols = self.raw_dataset.columns  # type: ignore

        value_counts_dict = {}
        for col in self.raw_dataset.columns:
            try:
                value_counts = self.raw_dataset[col].value_counts()
                print(value_counts)
                value_counts_dict[col] = value_counts
                print("-" * 50)
            except Exception:
                logger.warning(
                    "There was an issue printing the value counts of the `{col}` column which is of \
                    type {self.raw_dataset[col].dtype}. Please ensure it is of a numeric type that can \
                    be analysed using `pandas.Series.value_counts()`. Alternatively, specify the names \
                    of the columns you would like to view in the function call."
                )

        return value_counts_dict

    def _get_df_descr(
        self,
        cols: Optional[list[str]] = None,
    ) -> tuple[dict[str, pd.Series], pd.Series, int]:
        value_counts_dict = self._get_value_counts(cols=cols)
        null_summary: pd.Series = self.raw_dataset.isna().sum()
        num_duplicates: int = self.raw_dataset.duplicated().sum()

        return value_counts_dict, null_summary, num_duplicates
