# LoRA SFT Encoder Training

## Overview
The `train_lora.pbs` script is designed for training a LoRA (Low-Rank Adaptation) fine-tuned encoder model for multi-label text classification on GPU clusters using PBS (Portable Batch System) job scheduling. This model classifies reviews into four categories: advertisement, irrelevant content, non-visitor rants, and toxicity.

## Script Execution

The PBS script executes `sft_encoder.py`, which contains the complete training pipeline:

### What sft_encoder.py Does
1. **Data Loading**: Loads labeled training data from `final_huggingface_ds.csv`
2. **Model Initialization**: Creates a DistilBERT model with LoRA adapters for multi-label classification
3. **Dataset Creation**: Tokenizes text and prepares multi-label targets for 4 categories
4. **Training Loop**: Runs supervised fine-tuning with BCEWithLogitsLoss for multi-label classification
5. **Evaluation**: Computes accuracy, precision, recall, and F1-score per category
6. **Model Saving**: Saves both LoRA adapters and full model state
7. **Inference Testing**: Tests prediction on sample review text

### Training Process
- Splits data 80/20 for train/validation
- Uses BCEWithLogitsLoss for multi-label classification
- Applies gradient clipping and learning rate scheduling
- Saves best model based on validation loss
- Outputs detailed metrics for each classification category

## PBS Script Configuration

### System Requirements
- **GPU Cluster**: Designed for HPC environments with PBS job scheduler
- **GPU Requirements**: 8 GPUs (configured with `select=1:ngpus=8`)
- **Walltime**: 4 hours maximum
- **CUDA Version**: 12.2.2

### PBS Job Parameters
```bash
#PBS -P 71001002          # Project ID
#PBS -q ic102             # Queue name  
#PBS -N sft_lora_train    # Job name
#PBS -l select=1:ngpus=8  # Resource allocation
#PBS -l walltime=04:00:00 # 4-hour time limit
#PBS -j oe               # Join stdout/stderr
#PBS -o train_lora.log   # Output log file
```

## Data Requirements

### Primary Training Data
- **File**: `data/for_model/final_huggingface_ds.csv`
- **Format**: CSV with UTF-8-sig encoding
- **Required Columns**:
  - `text`: Review text content
  - `advertisement`: Binary label (0/1)
  - `relevance`: Binary label (0/1) - mapped to `irrelevant_content`
  - `rant`: Binary label (0/1) - mapped to `non_visitor_rant`  
  - `toxicity`: Binary label (0/1)

### Data Structure Example
```csv
text,spam,advertisement,relevance,rant,toxicity
"Business Name: Restaurant ABC...",0,0,1,0,0
```

## Model Architecture

### Base Model
- **Architecture**: DistilBERT (`distilbert-base-cased`)
- **Task**: Multi-label sequence classification (4 labels)
- **Max Sequence Length**: 512 tokens

### LoRA Configuration
- **Rank (r)**: 16
- **Alpha**: 32
- **Target Modules**: `["q_lin", "k_lin", "v_lin"]` (attention layers)
- **Dropout**: 0.1
- **Bias**: None

### Training Parameters
- **Batch Size**: 8
- **Learning Rate**: 1e-4
- **Weight Decay**: 0.01
- **Epochs**: 2
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau

## Some caveats

### 1. **GPU Cluster Dependency**
- Requires access to HPC cluster
- PBS job scheduler not available on personal machines
- 8-GPU requirement makes it unsuitable for consumer hardware

### 2. **Explanation of paramaters**
- Specific project ID (`71001002`) and queue (`ic102`) 
- CUDA module loading system specific to cluster environment
- Network-mounted storage paths

### 3. **Data Path Dependencies**
- Hardcoded relative paths assuming specific project structure
- Large dataset files not typically stored in version control
- Requires preprocessed training data in exact format

### 4. **Environment Setup**
- Custom Python virtual environment managed by `uv`
- Specific CUDA paths and module system
- Environment variables set for cluster-specific CUDA installation

## Usage Instructions

### Prerequisites
1. Access to PBS-managed GPU cluster
2. Project allocation with sufficient GPU hours
3. Training data in correct format and location
4. Python environment with required dependencies

### Submitting the Job
```bash
# Navigate to project root
cd /path/to/project

# Submit PBS job
qsub src/encoder/train_lora.pbs

# Monitor job status  
qstat -u $USER

# Check job output
cat train_lora.log
```

### Expected Outputs
- **LoRA Adapters**: Saved to current directory
- **Full Model State**: `lora_sft_encoder.pth`
- **Training Metrics**: `results/final_metrics.json`
- **Training Log**: `train_lora.log`

## Running Locally (Alternative)

For development/testing without PBS:

```bash
# Activate environment
source .venv/bin/activate
cd src/encoder

# Run training directly
python sft_encoder.py
```

**Note**: This requires GPU availability and may need parameter adjustments for memory constraints.

## Output Metrics
The model evaluates performance across four categories:
- Advertisement detection
- Irrelevant content filtering  
- Non-visitor rant identification
- Toxicity classification

Metrics include accuracy, precision, recall, and F1-score for each category.
