import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

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

"C:\Users\chunw\.cache\huggingface\hub\models--RunjiaChen--fasttext\snapshots\82ef8f92a35c845e9fdadcc057c65fe5a2b0e83c"