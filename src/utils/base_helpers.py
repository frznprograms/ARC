import time
import json
import torch
from typing import Any, Optional
import random
from loguru import logger

import pandas as pd

from src.eda.toxic_eda.jigsaw_preprocess import JigsawProcessor
from src.eda.toxic_eda.toxigen_preprocess import ToxigenProcessor
from src.eda.toxic_eda.twitter_preprocess import TwitterProcessor


@logger.catch(message="Unable to prepare the safety-related datasets.", reraise=True)
def prepare_safety_datasets(
    jigsaw_path: str,
    jigsaw_save_path: str,
    toxigen_save_path: str,
    twitter_path: str,
    twitter_save_path: str,
) -> pd.DataFrame:
    jigsaw = JigsawProcessor(file_path=jigsaw_path).preprocess(jigsaw_save_path)
    toxigen = ToxigenProcessor(file_path=None).preprocess(toxigen_save_path)
    twitter = TwitterProcessor(file_path=twitter_path).preprocess(twitter_save_path)

    combined_df = pd.concat([jigsaw, toxigen, twitter], axis=0)  # type: ignore
    combined_df.dropna(inplace=True)
    combined_df.drop_duplicates(inplace=True)

    return combined_df


@logger.catch(message="Unable to read json configuration.", reraise=True)
def read_json_safety_config(
    json_config_path: str = "src/configs/safety_config.json",
) -> dict[str, Any]:
    logger.info("Now reading json configuration for training...")

    with open(json_config_path, "r") as f:
        param_grid = json.load(f)
    # convert back lists to tuples for ngram range
    param_grid["features__tfidf__ngram_range"] = [
        tuple(x) for x in param_grid["features__tfidf__ngram_range"]
    ]
    # fix class_weight dicts
    new_class_weights = []
    for cw in param_grid["clf__class_weight"]:
        if isinstance(cw, dict):
            cw = {int(k): v for k, v in cw.items()}  # convert str -> int
        new_class_weights.append(cw)

    param_grid["clf__class_weight"] = new_class_weights
    logger.success("Read json configuration successfully.")

    return param_grid


@logger.catch(message="Unable to set_device.", reraise=True)
def set_device(device_type: str = "auto") -> str:
    device = None
    if device_type == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    else:
        if device_type == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                logger.warning(
                    f"Could not set device to {device_type}, defaulting to cpu instead."
                )
                device = "cpu"
        if device_type == "mps":
            if torch.mps.is_available():
                device = "mps"
            else:
                logger.warning(
                    f"Could not set device to {device_type}, defaulting to cpu instead."
                )
                device = "cpu"
        else:
            device = "cpu"

    if device is None:
        raise ValueError(f"Could not assign device of type {device_type}")

    return device


@logger.catch(message="Unable to set seed for this run/experiment.", reraise=True)
def set_seeds(seed_num: Optional[int], deterministic: bool = True) -> int:
    if seed_num is None:
        logger.warning("A seed was not detected. Setting seed to 42.")
        seed_num = 42
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed_num


def timed_execution(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_seconds = end - start

        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = elapsed_seconds % 60

        logger.info(
            f"Function executed in: {hours} hours, {minutes} minutes, {seconds:.3f} seconds"
        )
        return result

    return wrapper
