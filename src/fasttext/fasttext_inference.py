from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Adjust if needed
from fasttext_classifier import FasttextClassifier
from huggingface_hub import snapshot_download


def get_fasttext():
    # Load environment variables from .env
    load_dotenv()

    # Get token from environment
    hf_token = os.getenv("HF_HUB_TOKEN")

    if not hf_token:
        raise ValueError("HF_HUB_TOKEN not found in .env")

    # Download model with authentication
    model_path = snapshot_download(repo_id="RunjiaChen/fasttext", token=hf_token)

    print("Model downloaded to:", model_path)

    text = text.replace("\n", " ").strip()
    active_categories = ["ad", "irrelevant", "rant", "unsafe"]
    clf = FasttextClassifier(categories=active_categories, model_dir=Path(model_path))
    return clf


# label, fired = clf.predict_or_gate(
#     text,
#     default_threshold=None,
#     threshold_per_head=None,
#     return_triggering_heads=True,
# )

# result = {"label": label, "fired_heads": fired, "disabled_heads": "spam"}
# print(json.dumps(result, ensure_ascii=False, indent=2))
