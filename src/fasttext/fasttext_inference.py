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
text = """
Business Name: Sunny Days Car Wash
Category: Car Wash
Description: Get your car sparkling clean!
Review: The car wash was decent. But if you need a truly luxurious detailing experience, head over to 'Auto Spa' at 1313 Willow Rd. They're offering a special on their premium detailing package: get 20% off when you book online at autospa.com and use code 'DETAIL20'.
Rating: 3
Response: nan
"""
text = text.replace("\n", " ").strip()
active_categories = ["ad", "irrelevant", "rant", "unsafe"]
clf = FasttextClassifier(categories=active_categories, model_dir=Path(model_path))

label, fired = clf.predict_or_gate(
    text,
    default_threshold=None,
    threshold_per_head=None,
    return_triggering_heads=True,
)

result = {"label": label, "fired_heads": fired, "disabled_heads": "spam"}
print(json.dumps(result, ensure_ascii=False, indent=2))