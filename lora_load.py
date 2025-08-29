import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

# Check if local LoRA weights exist, otherwise download from HuggingFace
local_lora_path = ""
if not os.path.exists(local_lora_path):
    from huggingface_hub import hf_hub_download
    lora_weights_path = "dolphin-in-teal-lake/sft-encoder" 
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
if os.path.exists("lora_sft_encoder.pth"):
    state_dict = torch.load("lora_sft_encoder.pth", map_location='cpu')
    
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

test_text = '''MY WIFE LIKES TO GET HER NAILS DONE AND THERE'S A PLACE ON MOTHER GASTON CALL SUN NAILS AND THEIR PRICES ARE VERY REASONABLE BUT LUCKY NAIL 
            IS A LITTLE EXPENSIVE LOVE YOU TOO EXPENSIVE FOR MY TASTE FOR THE TYPE OF WORK THEY DO... SO I WOULD RECOMMEND A PLACE LIKE SUN NAILS EVEN IF IT'S OUT OF YOUR WAY 
            A LITTLE BIT IT'S WORTH SAVING A COUPLE OF DOLLARS OR I WILL SAY A LITTLE MORE THAN A COUPLE OF DOLLARS IF YOU'RE GETTING MORE THAN JUST A NAIL IS DONE...
            '''
            

# Tokenize
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.sigmoid(outputs.logits)
    predictions = (probabilities > 0.5).int()

print("Predictions:", predictions)  # [advertisement, irrelevant_content, non_visitor_rant, toxicity]
print("Probabilities:", probabilities)
