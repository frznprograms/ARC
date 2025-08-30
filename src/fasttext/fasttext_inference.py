from __future__ import annotations

import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import argparse
import json
from pathlib import Path
from typing import Dict

# Adjust if needed
from fasttext_classifier import FasttextClassifier

def get_fasttext_model():

    # Load environment variables from .env
    load_dotenv()

    # Get token from environment
    hf_token = os.getenv("HF_HUB_TOKEN")

    if not hf_token:
        raise ValueError("HF_HUB_TOKEN not found in .env")

    # Download model with authentication
    model_path = snapshot_download(
        repo_id="RunjiaChen/fasttext",
        token=hf_token
    )

    print("Model downloaded to:", model_path)

    text = text.replace("\n", " ").strip()
    active_categories = ["ad", "irrelevant", "rant", "unsafe"]
    clf = FasttextClassifier(categories=active_categories, model_dir=Path(model_path))
    return clf

