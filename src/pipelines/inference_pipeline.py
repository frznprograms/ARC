import os
from dataclasses import dataclass
from typing import Any

import joblib
import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()
# Get token from environment
hf_token = os.getenv("HF_HUB_TOKEN")

if not hf_token:
    logger.error("HF_HUB_TOKEN not found in .env")


@dataclass
class InferencePipeline:
    safety_model_path: str
    fasttext_model_path: str
    encoder_model_path: str = "lora_sft_encoder.pth"

    def __post_init__(self):
        self.safety_model = joblib.load(self.safety_model_path)
        self.fasttext_model = snapshot_download(
            repo_id="RunjiaChen/fasttext", token=hf_token
        )
        self.encoder = self._load_encoder()

    def run_inference(self, review: dict[str, Any]):
        safe_value = self.safety_model.predict(review)
        pred_strength = self.safety_model.predict_proba(review)[:, 1]
        if safe_value > 0:
            logger.warning(
                f"The review did not pass the saftey check with probabilit {pred_strength}, \
                    and has been flagged for rejection."
            )
            return safe_value

        self.fasttext_model.predict(
            review
        )  # TODO: implement prediction logic for fasttext

        # encoder section, stage 3
        inputs = self.tokenizer(
            review, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).int()

        # since prediction is just one value
        final_pred = bool(preds[0])
        logger.info(
            f"Final prediction after 3 stages is {final_pred} with probability {probs}."
        )
        return 3

    def _load_encoder(self):
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
        logger.success("Loaded base model.")

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
