import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List, Optional
import json
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


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
    def __init__(self, model_name: str = "distilbert-base-uncased", lora_rank: int = 16, lora_alpha: float = 32):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model for multi-label classification (4 labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=4,
            problem_type="multi_label_classification"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_lin", "k_lin", "v_lin"],  # DistilBERT attention layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        self.label_names = ['advertisement', 'irrelevant_content', 'non_visitor_rant', 'toxicity']
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def predict_review_classification(self, review_prompt: str, threshold: float = 0.5) -> ReviewClassificationResult:
        """
        Predict classification for a single review prompt
        """
        self.model.eval()
        
        encoding = self.tokenizer(
            review_prompt,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            logits = outputs.logits.squeeze()
            probabilities = torch.sigmoid(logits).numpy()
            
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
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Print trainable parameter info
        self.model.model.print_trainable_parameters()
        
        # Only optimize trainable parameters (LoRA adapters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
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
            
            outputs = self.model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            
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
                
                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                logits = outputs.logits
                predictions = (torch.sigmoid(logits) > 0.5).float()
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
