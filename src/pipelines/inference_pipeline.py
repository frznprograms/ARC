import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

from src.fasttext.fasttext_classifier import FasttextClassifier
from src.utils.base_helpers import suppress_stdout_stderr, timed_execution

load_dotenv()
# Get token from environment
hf_token = os.getenv("HF_HUB_TOKEN")

if not hf_token:
    logger.warning("HF_HUB_TOKEN not found in environment.")


@dataclass
class InferencePipeline:
    """
    A pipeline for running multi-stage inference on reviews using various models.

    Attributes:
        safety_model_path (str): Path to the safety model file.
        encoder_model_path (str): Path to the encoder model file.
    """

    safety_model_path: str = "models/safety-model-test.pkl"
    encoder_model_path: str = "lora_sft_encoder.pth"

    def __post_init__(self):
        """
        Initializes the pipeline by loading the models and setting up configurations.
        """
        hf_logging.set_verbosity_error()
        logger.info("Loading models...")
        self.safety_model = joblib.load(self.safety_model_path)
        logger.success("Loaded safety model for Stage 1 checks.")

        with suppress_stdout_stderr():
            fasttext_model_path = snapshot_download(
                repo_id="RunjiaChen/fasttext", token=hf_token
            )
            active_categories = ["ad", "irrelevant", "rant", "unsafe"]
            self.fasttext_model = FasttextClassifier(
                categories=active_categories, model_dir=Path(fasttext_model_path)
            )
        logger.success("Loaded fasttext heads for Stage 2 checks.")

        self.encoder = self._load_encoder()
        logger.success("Loaded encoder model for Stage 3 checks.")

    @timed_execution
    @logger.catch(message="Unable to complete inference for review.", reraise=True)
    def run_inference(
        self,
        review_and_metdata: dict[str, Any],
        default_threshold: float = 0.7,
    ) -> int:
        """
        Runs the inference pipeline on a given review and metadata.

        Args:
            review_and_metdata (dict[str, Any]): A dictionary containing the review and its metadata.
                Expected keys include 'review', 'name', 'category', 'description', and 'rating'.
            default_threshold (float): The default threshold for classification decisions. Defaults to 0.7.

        Returns:
            int: The stage at which the review was rejected or the final stage if accepted.

        Raises:
            ValueError: If the review is empty or missing.
        """
        stage = 1
        review = review_and_metdata.get("review", None)
        if review is None:
            logger.error("Review is empty.")

        if isinstance(review, str):
            review = [review]
        safe_value = self.safety_model.predict(review)
        pred_strength = self.safety_model.predict_proba(review)[:, 1]
        if safe_value > 0:
            logger.warning(
                f"The review did not pass the saftey check with probability {pred_strength}, and has therefore been rejected."
            )
            return stage

        stage += 1
        # fasttext section, stage 2
        prompt = f"""
            Business Name: {review_and_metdata["name"]}
            Category: {review_and_metdata["category"]}
            Description: {review_and_metdata["description"]}
            Review: {review_and_metdata["review"]}
            Rating: {review_and_metdata["rating"]}
        """
        prompt = prompt.replace("\n", "").strip()

        label, fired = self.fasttext_model.predict_or_gate(
            prompt,
            default_threshold=default_threshold,
            return_triggering_heads=True,
        )
        if label == "bad":
            logger.warning(
                f"The review has been rejected by fasttext heads, where the fired heads are: {fired}."
            )
            return stage
        else:
            logger.success(
                f"Review was accepted at stage {stage} and passed all policies!"
            )

        # encoder section, stage 3
        stage += 1
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).int()

        # since prediction is just one value
        final_pred_val = torch.max(preds, dim=0)
        final_pred_accept = "Accept" if final_pred_val == 0 else "Reject"
        if final_pred_accept:
            logger.success(
                f"Review was accepted at stage {stage} and passed all policies!"
            )
        else:
            logger.info(
                f"Final prediction after 3 stages is {final_pred_accept} with probability {probs}."
            )
        return stage

    @logger.catch(message="Unable to load encoder.", reraise=True)
    def _load_encoder(self):
        """
        Loads the encoder model with LoRA configuration.

        Returns:
            torch.nn.Module: The loaded encoder model.

        Raises:
            FileNotFoundError: If the encoder model file is not found.
        """
        if not os.path.exists(self.encoder_model_path):
            from huggingface_hub import hf_hub_download

            lora_weights_path = hf_hub_download(
                repo_id="dolphin-in-teal-lake/sft-encoder",
                filename="lora_sft_encoder.pth",
            )
        else:
            lora_weights_path = self.encoder_model_path

        # load base model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-cased",
            num_labels=4,
            problem_type="multi_label_classification",
        )
        logger.success("Loaded base encoder model.")

        # create LORA model structure first
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_lin", "k_lin", "v_lin"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        logger.success("Loaded LoRA config.")
        model = get_peft_model(base_model, lora_config)

        # load_weights
        if os.path.exists(lora_weights_path):
            state_dict = torch.load(lora_weights_path, map_location="cpu")

            classifier_keys = []
            lora_keys = []
            for key in sorted(state_dict.keys()):
                if "classifier" in key or "pre_classifier" in key:
                    classifier_keys.append(key)
                if "lora" in key:
                    lora_keys.append(key)

            # create a mapped state dict
            mapped_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key[6:]
                    mapped_state_dict[new_key] = value
                else:
                    mapped_state_dict[key] = value

            model.load_state_dict(mapped_state_dict, strict=False)
            logger.success("Loaded ALL weights from .pth file with key mapping")

        model.eval()

        return model


if __name__ == "__main__":
    """
    Main entry point for running the inference pipeline on sample reviews.
    """
    ip = InferencePipeline(safety_model_path="models/safety-model-test.pkl")

    reviews = []
    for i in range(1, 4):
        with open(f"data/for_model/review_{i}.json", "r") as f:
            reviews.append(json.load(f))

    for review in reviews:
        ip.run_inference(review)
