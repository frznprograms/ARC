import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import numpy as np
from typing import Dict, List, Optional
import json
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import math


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


@dataclass
class ReviewClassificationResult:
    advertisement: int
    irrelevant_content: int
    non_visitor_rant: int
    toxicity: int
    confidence_scores: Dict[str, float]


class ReviewClassificationDataset(Dataset):
    def __init__(self, prompts: List[str], labels: List[Dict], tokenizer, max_length: int = 512):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label_dict = self.labels[idx]
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert labels to tensor
        label_tensor = torch.tensor([
            int(label_dict.get('advertisement', 0)),
            int(label_dict.get('irrelevant_content', 0)),
            int(label_dict.get('non_visitor_rant', 0)),
            int(label_dict.get('toxicity', 0))
        ], dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_tensor
        }


class ReviewSFTEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout_rate: float = 0.3, lora_rank: int = 16, lora_alpha: float = 16):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        hidden_size = self.backbone.config.hidden_size
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters to middle transformer layers
        self.lora_adapters = nn.ModuleDict()
        
        # For DistilBERT, apply LoRA to middle layers (layers 2-4 out of 6 layers)
        target_layers = [2, 3, 4]
        
        for layer_idx in target_layers:
            layer = self.backbone.transformer.layer[layer_idx]
            
            # Apply LoRA to query, key, value projections in attention
            attention = layer.attention
            q_proj = attention.q_lin
            k_proj = attention.k_lin  
            v_proj = attention.v_lin
            
            # Create LoRA adapters for Q, K, V projections
            self.lora_adapters[f'layer_{layer_idx}_q'] = LoRALayer(
                q_proj.in_features, q_proj.out_features, lora_rank, lora_alpha
            )
            self.lora_adapters[f'layer_{layer_idx}_k'] = LoRALayer(
                k_proj.in_features, k_proj.out_features, lora_rank, lora_alpha
            )
            self.lora_adapters[f'layer_{layer_idx}_v'] = LoRALayer(
                v_proj.in_features, v_proj.out_features, lora_rank, lora_alpha
            )
            
            # Apply LoRA to feed-forward layers
            ffn = layer.ffn
            lin1 = ffn.lin1
            lin2 = ffn.lin2
            
            self.lora_adapters[f'layer_{layer_idx}_ffn1'] = LoRALayer(
                lin1.in_features, lin1.out_features, lora_rank, lora_alpha
            )
            self.lora_adapters[f'layer_{layer_idx}_ffn2'] = LoRALayer(
                lin2.in_features, lin2.out_features, lora_rank, lora_alpha
            )
        
        # Multi-task classification head (trainable)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 4),  # 4 classification tasks
            nn.Sigmoid()
        )
        
        self.label_names = ['advertisement', 'irrelevant_content', 'non_visitor_rant', 'toxicity']
        
    def forward(self, input_ids, attention_mask):
        # Use the standard backbone forward pass
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states
        
        # Apply LoRA adapters to middle layers
        x = hidden_states[-1]  # Start with final layer output
        
        # Apply LoRA to specific middle layers (layers 2-4)
        for i in [2, 3, 4]:
            layer_hidden = hidden_states[i+1]  # hidden_states[0] is embeddings, so layer i is at index i+1
            
            # Apply LoRA to attention projections
            q_lora = self.lora_adapters[f'layer_{i}_q'](layer_hidden)
            k_lora = self.lora_adapters[f'layer_{i}_k'](layer_hidden)
            v_lora = self.lora_adapters[f'layer_{i}_v'](layer_hidden)
            
            # Apply LoRA to FFN
            ffn1_lora = self.lora_adapters[f'layer_{i}_ffn1'](layer_hidden)
            ffn2_lora = self.lora_adapters[f'layer_{i}_ffn2'](ffn1_lora)
            
            # Add LoRA outputs to final representation
            x = x + q_lora + k_lora + v_lora + ffn2_lora
        
        # Use CLS token representation
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
    
    def predict_review_classification(self, review_prompt: str, threshold: float = 0.5) -> ReviewClassificationResult:
        """
        Predict classification for a single review prompt
        """
        self.eval()
        
        encoding = self.tokenizer(
            review_prompt,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = self.forward(
                encoding['input_ids'],
                encoding['attention_mask']
            ).squeeze()
            
        probabilities = logits.numpy() if isinstance(logits, torch.Tensor) else logits
        
        # Convert probabilities to binary predictions
        predictions = (probabilities > threshold).astype(int)
        
        confidence_scores = {
            label: float(prob) for label, prob in zip(self.label_names, probabilities)
        }
        
        return ReviewClassificationResult(
            advertisement=int(predictions[0]),
            irrelevant_content=int(predictions[1]),
            non_visitor_rant=int(predictions[2]),
            toxicity=int(predictions[3]),
            confidence_scores=confidence_scores
        )
    
    def generate_response_string(self, review_prompt: str, threshold: float = 0.5) -> str:
        """
        Generate the expected response string format for SFT training
        """
        result = self.predict_review_classification(review_prompt, threshold)
        
        response_dict = {
            "advertisement": result.advertisement,
            "irrelevant_content": result.irrelevant_content,
            "non_visitor_rant": result.non_visitor_rant,
            "toxicity": result.toxicity
        }
        
        return json.dumps(response_dict)


class ReviewDataProcessor:
    def __init__(self):
        self.label_names = ['spam', 'advertisement', 'irrelevant_content', 'non_visitor_rant', 'toxicity']
    
    def load_training_data_from_json(self, json_dir: str) -> tuple[List[str], List[Dict]]:
        """
        Load training data from JSON files generated by the label_gen process
        """
        prompts = []
        labels = []
        
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(json_dir, filename)
                
                with open(filepath, 'r') as f:
                    batch_data = json.load(f)
                    
                for item in batch_data:
                    # Extract the prompt (this would need to be reconstructed from job_id)
                    # For now, we'll assume the prompt is stored or can be retrieved
                    job_id = item['job_id']
                    
                    # Convert string labels to integers
                    label_dict = {}
                    for label_name in self.label_names:
                        if label_name in item:
                            label_dict[label_name] = int(item[label_name])
                        else:
                            label_dict[label_name] = 0
                    
                    labels.append(label_dict)
                    
        return prompts, labels
    
    def load_labeled_data_from_csv(self, csv_path: str) -> tuple[List[str], List[Dict]]:
        """
        Load labeled training data directly from CSV
        """
        df = pd.read_csv(csv_path)
        prompts = []
        labels = []
        
        for _, row in df.iterrows():
            prompts.append(row['text'])
            
            # Map CSV columns to our label format
            label_dict = {
                'advertisement': int(row.get('advertisement', 0)),
                'irrelevant_content': 1 - int(row.get('relevance', 1)),  # relevance=0 means irrelevant=1
                'non_visitor_rant': int(row.get('rant', 0)),
                'toxicity': int(row.get('toxicity', 0))
            }
            labels.append(label_dict)
            
        return prompts, labels
    
    def create_prompt_response_pairs(self, df: pd.DataFrame, labeled_data: List[Dict]) -> tuple[List[str], List[str]]:
        """
        Create prompt-response pairs for SFT training
        """
        prompts = []
        responses = []
        
        # Create a mapping from job_id to labels
        label_map = {item['job_id']: item for item in labeled_data}
        
        for _, row in df.iterrows():
            job_id = row['id']
            
            if job_id in label_map:
                prompt = row['user_message']
                
                # Create response string
                label_dict = label_map[job_id]
                response_dict = {}
                
                for label_name in self.label_names:
                    response_dict[label_name] = int(label_dict.get(label_name, 0))
                
                response_string = json.dumps(response_dict)
                
                prompts.append(prompt)
                responses.append(response_string)
                
        return prompts, responses
    


class ReviewSFTTrainer:
    def __init__(self, model: ReviewSFTEncoder, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        
        # Only optimize LoRA parameters and classifier
        trainable_params = []
        
        # Add LoRA adapter parameters
        for name, param in self.model.lora_adapters.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                
        # Add classifier parameters
        for name, param in self.model.classifier.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        print(f"Training {len(trainable_params)} parameters (LoRA + classifier only)")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {trainable_param_count:,} / {total_params:,} ({100 * trainable_param_count / total_params:.2f}%)")
        
        self.optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        
        metrics = {}
        categories = ['advertisement', 'irrelevant_content', 'non_visitor_rant', 'toxicity']
        
        for i, category in enumerate(categories):
            if all_labels[:, i].sum() > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels[:, i], all_predictions[:, i], average='binary', zero_division=0
                )
                accuracy = accuracy_score(all_labels[:, i], all_predictions[:, i])
                
                metrics[f'{category}_accuracy'] = accuracy
                metrics[f'{category}_precision'] = precision
                metrics[f'{category}_recall'] = recall
                metrics[f'{category}_f1'] = f1
        
        metrics['avg_loss'] = total_loss / len(dataloader)
        metrics['overall_accuracy'] = np.mean([metrics.get(f'{cat}_accuracy', 0) for cat in categories])
        
        return metrics
    
    def train_model(self, train_dataloader: DataLoader, val_dataloader: DataLoader, 
                   num_epochs: int = 10, save_path: str = None) -> Dict:
        """
        Train the SFT decoder model
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print("Starting SFT Encoder training...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_dataloader)
            val_metrics = self.evaluate(val_dataloader)
            val_loss = val_metrics['avg_loss']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Overall Accuracy: {val_metrics['overall_accuracy']:.4f}")
            
            # Print F1 scores for each category
            categories = ['advertisement', 'irrelevant_content', 'non_visitor_rant', 'toxicity']
            for cat in categories:
                f1_key = f'{cat}_f1'
                if f1_key in val_metrics:
                    print(f"{cat.title()} F1: {val_metrics[f1_key]:.4f}")
            
            print("-" * 60)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Best model saved to {save_path}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_metrics': val_metrics
        }


def main():
    """
    Main function for SFT encoder training
    """
    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = ReviewSFTEncoder()
    processor = ReviewDataProcessor()
    trainer = ReviewSFTTrainer(model, device)
    
    # Load labeled training data from CSV
    print("Loading training data...")
    prompts, labels = processor.load_labeled_data_from_csv("final_hf_ds.csv")
    print(f"Loaded {len(prompts)} labeled samples from CSV")
    
    # Split data into train/validation sets
    train_prompts, val_prompts, train_labels, val_labels = train_test_split(
        prompts, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_prompts)}")
    print(f"Validation samples: {len(val_prompts)}")
    
    # Create datasets
    train_dataset = ReviewClassificationDataset(train_prompts, train_labels, model.tokenizer)
    val_dataset = ReviewClassificationDataset(val_prompts, val_labels, model.tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Train model
    print("\nStarting training...")
    results = trainer.train_model(
        train_dataloader, 
        val_dataloader, 
        num_epochs=10,
        save_path="review_sft_encoder_lora.pth"
    )
    
    # Test inference with a sample
    print("\nTesting inference...")
    test_prompt = "Business Name: Test Restaurant\nCategory: Restaurant\nReview: This place is absolutely terrible! The staff are idiots!\nResponse: None"
    
    result = model.predict_review_classification(test_prompt)
    print(f"Prediction: {result}")
    
    response_string = model.generate_response_string(test_prompt)
    print(f"Response string: {response_string}")


if __name__ == "__main__":
    main()
