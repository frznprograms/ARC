from dataclasses import dataclass
from datasets import load_dataset
import pandas as pd
from typing import Union
from loguru import logger
from pathlib import Path

from eda.toxic_eda.base_preprocess import BaseProcessor


@dataclass
class ToxigenProcessor(BaseProcessor):

    def preprocess(self, save_path: Union[str, Path]):
        logger.info("Starting preprocessing...")

        gen_intermediate = self._compile_dataset()
        gen_cleaned = self._clean_compiled_dataset(df=gen_intermediate)
        annotated_cleaned = self._clean_annotated_dataset()

        cleaned = pd.concat([gen_cleaned, annotated_cleaned], axis=0)  # type: ignore

        self.save_data(data=cleaned, path=save_path)

        logger.success("Preprocessed Toxigen dataset successfully.")

    def read_hf_data(self) -> None:
        gen_data = load_dataset("toxigen/toxigen-data", name="train")[
            "train"
        ].to_pandas()  # type: ignore
        annotated_data = load_dataset("toxigen/toxigen-data", name="annotated")[
            "train"
        ].to_pandas()  # type: ignore

        self.gen_data, self.annotated_data = gen_data, annotated_data

    def _compile_dataset(self):
        prompts, generations, labels = (
            self.gen_data["prompt"],  # type: ignore
            self.gen_data["generation"],  # type: ignore
            self.gen_data["prompt_label"],  # type: ignore
        )
        new_entries, new_labels = [], []
        n = len(self.raw_dataset)
        for i in range(n):
            full_prompt, generation, label = (
                prompts.iloc[i],
                generations.iloc[i],
                labels.iloc[i],
            )
            sentences: list[str] = full_prompt.split("\\n")
            for sentence in sentences:
                new_entries.append(sentence)
                new_labels.append(label)
            # add the generation too
            new_entries.append(generation)
            new_labels.append(label)

        return pd.DataFrame({"text": new_entries, "unsafe_label": new_labels})

    def _clean_compiled_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove leading dash and surrounding whitespace
        df["text"] = df["text"].str.lstrip("-").str.strip()
        # Keep only rows where text contains at least one alphabet (A–Z or a–z)
        df = df[df["text"].str.contains(r"[A-Za-z]", na=False)]  # type: ignore

        return df.reset_index(drop=True)

    def _clean_annotated_dataset(self):
        toxic_annotated_data = self.annotated_data[  # type: ignore
            self.annotated_data["toxicity_human"] > 2.5  # type: ignore
        ]
        toxic_annotated_data = toxic_annotated_data[["text", "toxicity_human"]]
        toxic_annotated_data["unsafe_label"] = 1

        safe_annotated_data = self.annotated_data[  # type: ignore
            self.annotated_data["toxicity_human"] <= 2  # type: ignore
        ]
        safe_annotated_data = safe_annotated_data[["text", "toxicity_human"]]
        safe_annotated_data["unsafe_label"] = 0

        self.annotated_data = safe_annotated_data
