import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

# Check if local LoRA weights exist, otherwise download from HuggingFace
local_lora_path = "lora_sft_encoder.pth"
if not os.path.exists(local_lora_path):
    from huggingface_hub import hf_hub_download
    lora_weights_path = hf_hub_download(
        repo_id="dolphin-in-teal-lake/sft-encoder",
        filename="lora_sft_encoder.pth"
    )
else:
    lora_weights_path = local_lora_path

# Load base model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-cased", 
    num_labels=4,
    problem_type="multi_label_classification"
)

# Create LoRA model structure first
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_lin", "k_lin", "v_lin"],
    lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS
)
model = get_peft_model(base_model, lora_config)


# Load weights
if os.path.exists(lora_weights_path):
    state_dict = torch.load(lora_weights_path, map_location='cpu')
    
    print("Keys in state_dict:")
    classifier_keys = []
    lora_keys = []
    for key in sorted(state_dict.keys()):
        if 'classifier' in key or 'pre_classifier' in key:
            classifier_keys.append(key)
        if 'lora' in key:
            lora_keys.append(key)
    
    # Create a mapped state dict
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[6:]  
            mapped_state_dict[new_key] = value
        else:
            mapped_state_dict[key] = value
    
    model.load_state_dict(mapped_state_dict, strict=False)
    print("Loaded ALL weights from .pth file with key mapping")

model.eval()

test_text = '''Business Name: Happy Nails Salon
Category: Hair salon
Description: Offering nail services, hair cuts and styling
Review: My nails look like a 5-year-old did them, the girl was rude, and the place was filthy. I'm seriously considering reporting them. If you value your money, don't even think about going there!
Rating: 1
Response: nan
            '''
            

# Tokenize
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)
    predictions = (probabilities > 0.5).int()

print("Predictions:", predictions)  
print("Probabilities:", probabilities)
