# FastText Classifier Usage Guide

This guide explains how to use the FastText multi-head classifier for content moderation. The workflow consists of three main steps:

1. **Data Conversion**: Convert CSV data to FastText format
2. **Model Training**: Train FastText models for each category
3. **Inference**: Run predictions using the trained models

## Overview

The FastText classifier uses a hard OR-gate approach where if ANY category head predicts positive (bad content), the overall result is "bad", otherwise it's "good". The system supports multiple categories: spam, ad, irrelevant, rant, and unsafe.

---

## Step 1: Data Conversion (`convert_csv_to_fasttext.py`)

Converts labeled review samples from CSV format into FastText `.txt` files with automatic train/eval split.

### Input Requirements

Your CSV file must have these columns:
- `spam`, `ad`, `irrelevant`, `rant`, `unsafe` (values: 0=negative/good, 1=positive/bad)
- `text` (the content to classify)

### Usage

```bash
# Basic usage
python src/fasttext/convert_csv_to_fasttext.py data/reviews.csv

# Specify custom output directory
python src/fasttext/convert_csv_to_fasttext.py data/reviews.csv -o custom_output_dir
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `input_path` | str | Yes | - | Path to CSV file with labeled samples |
| `-o, --out-dir` | str | No | `fasttext_out` | Directory to write output files |

### Example Commands

```bash
# Convert reviews.csv to FastText format in default directory
python src/fasttext/convert_csv_to_fasttext.py data/labeled_reviews.csv

# Convert with custom output directory
python src/fasttext/convert_csv_to_fasttext.py data/labeled_reviews.csv --out-dir fasttext_data

# Convert large dataset
python src/fasttext/convert_csv_to_fasttext.py data/10k_reviews.csv -o training_data
```

### Output Structure

```
out_dir/
├── train/
│   ├── spam.txt
│   ├── ad.txt
│   ├── irrelevant.txt
│   ├── rant.txt
│   └── unsafe.txt
└── eval/
    ├── spam.txt
    ├── ad.txt
    ├── irrelevant.txt
    ├── rant.txt
    └── unsafe.txt
```

---

## Step 2: Model Training (`train_fasttext_classifier.py`)

Trains FastText models for each category using the converted data files.

### Usage

```bash
# Basic training
python src/fasttext/train_fasttext_classifier.py --data-dir fasttext_out/train --out-dir models

# Training with custom hyperparameters
python src/fasttext/train_fasttext_classifier.py --data-dir fasttext_out/train --out-dir models --lr 0.1 --epoch 25 --dim 200
```

### Arguments

#### Required Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `--data-dir` | str | Directory containing `<category>.txt` training files |
| `--out-dir` | str | Directory to save `<category>.bin` model files |

#### Optional Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--categories` | list | `[spam, ad, irrelevant, rant, unsafe]` | Categories to train |
| `--preload-dir` | str | None | Directory with existing models for warm-start |
| `--skip-missing` | flag | False | Skip categories with missing training files |
| `--positive-label` | str | `__label__pos` | Positive class label for each head |

#### Hyperparameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | 0.5 | Learning rate |
| `--epoch` | int | 10 | Number of training epochs |
| `--dim` | int | 100 | Word vector dimension |
| `--minn` | int | 2 | Minimum character n-gram length |
| `--maxn` | int | 5 | Maximum character n-gram length |
| `--wordNgrams` | int | 2 | Word n-gram length |

#### Autotune Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--autotune` | flag | False | Enable FastText autotune |
| `--autotune-duration` | int | 60 | Seconds to spend autotuning |
| `--valid-suffix` | str | `_valid.txt` | Validation file suffix |

### Example Commands

```bash
# Basic training with default parameters
python src/fasttext/train_fasttext_classifier.py \
  --data-dir fasttext_out/train \
  --out-dir models

# Training with custom hyperparameters
python src/fasttext/train_fasttext_classifier.py \
  --data-dir fasttext_out/train \
  --out-dir models \
  --lr 0.1 \
  --epoch 25 \
  --dim 200 \
  --wordNgrams 3

# Training only specific categories
python src/fasttext/train_fasttext_classifier.py \
  --data-dir fasttext_out/train \
  --out-dir models \
  --categories spam ad unsafe

# Training with autotune (requires validation files)
python src/fasttext/train_fasttext_classifier.py \
  --data-dir fasttext_out/train \
  --out-dir models \
  --autotune \
  --autotune-duration 120

# Warm-start from existing models
python src/fasttext/train_fasttext_classifier.py \
  --data-dir fasttext_out/train \
  --out-dir new_models \
  --preload-dir old_models \
  --epoch 5

# Training with custom validation suffix
python src/fasttext/train_fasttext_classifier.py \
  --data-dir fasttext_out/train \
  --out-dir models \
  --autotune \
  --valid-suffix _dev.txt

# Skip missing categories instead of failing
python src/fasttext/train_fasttext_classifier.py \
  --data-dir fasttext_out/train \
  --out-dir models \
  --skip-missing
```

---

## Step 3: Inference (`inference.py`)

Run predictions using the trained models with a hard OR-gate approach.

### Usage

```bash
# Basic inference
python src/fasttext/inference.py --model-dir models --text "Your text here"

# Inference with custom thresholds
python src/fasttext/inference.py --model-dir models --text "Your text here" --default-threshold 0.7
```

### Arguments

#### Required Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `--model-dir` | str | Directory containing `<category>.bin` model files |
| `--text` | str | Input text to classify |

#### Optional Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--default-threshold` | float | None | Default probability threshold for all heads |
| `--per-head-thresholds` | str | None | Per-head thresholds (format: `cat1:0.7,cat2:0.6`) |
| `--<category>` | str | true | Enable/disable specific category heads |

### Example Commands

```bash
# Basic inference
python src/fasttext/inference.py \
  --model-dir models \
  --text "This is a great product, buy now!"

# Inference with custom threshold
python src/fasttext/inference.py \
  --model-dir models \
  --text "Click here for amazing deals!" \
  --default-threshold 0.8

# Inference with per-head thresholds
python src/fasttext/inference.py \
  --model-dir models \
  --text "Limited time offer!" \
  --per-head-thresholds "spam:0.9,ad:0.7,unsafe:0.5"

# Disable specific category heads
python src/fasttext/inference.py \
  --model-dir models \
  --text "This product is terrible" \
  --spam false \
  --rant true

# Enable only specific heads
python src/fasttext/inference.py \
  --model-dir models \
  --text "Free money now!" \
  --spam true \
  --ad true \
  --irrelevant false \
  --rant false \
  --unsafe false

# Real-world content examples
python src/fasttext/inference.py \
  --model-dir models \
  --text "URGENT: Your account will be suspended unless you click this link immediately!"

python src/fasttext/inference.py \
  --model-dir models \
  --text "I absolutely hate this company and their terrible customer service" \
  --default-threshold 0.6

python src/fasttext/inference.py \
  --model-dir models \
  --text "The weather is nice today" \
  --per-head-thresholds "irrelevant:0.3"
```

### Output Format

The inference script outputs JSON with the following structure:

```json
{
  "label": "bad",
  "fired_heads": ["spam", "ad"],
  "disabled_heads": []
}
```

- `label`: Overall classification ("good" or "bad")
- `fired_heads`: List of category heads that triggered (predicted positive)
- `disabled_heads`: List of heads that were disabled via flags

---

## Complete Workflow Example

Here's a complete example from CSV to inference:

```bash
# 1. Convert CSV data to FastText format
python src/fasttext/convert_csv_to_fasttext.py \
  data/reviews.csv \
  --out-dir training_data

# 2. Train models with custom hyperparameters
python src/fasttext/train_fasttext_classifier.py \
  --data-dir training_data/train \
  --out-dir models \
  --lr 0.1 \
  --epoch 20 \
  --dim 150

# 3. Run inference on new content
python src/fasttext/inference.py \
  --model-dir models \
  --text "Amazing discount! Click now before it expires!" \
  --default-threshold 0.75
```



## Tips and Best Practices

1. **Data Quality**: Ensure your CSV data is well-balanced across categories
2. **Hyperparameter Tuning**: Use `--autotune` for automatic hyperparameter optimization
3. **Thresholds**: Adjust thresholds based on your precision/recall requirements
4. **Category Selection**: Disable irrelevant heads during inference to reduce false positives
5. **Model Updates**: Use `--preload-dir` for incremental training when you get new data


---

### FastText Model Evaluation Results

Our FastText multi-head classifier was evaluated on a test set with the following performance metrics:

#### Overall Performance Summary

| Category | Precision | Recall | F1-Score* | Total Samples | Positive Samples |
|----------|-----------|--------|-----------|---------------|------------------|
| **Spam** | 100.00% | 99.73% | 99.86% | 2,886 | 734 |
| **Ad** | 99.60% | 98.43% | 99.01% | 2,886 | 255 |
| **Irrelevant** | 99.53% | 94.17% | 96.78% | 2,886 | 223 |
| **Rant** | 89.36% | 85.71% | 87.50% | 2,886 | 49 |
| **Unsafe** | 95.43% | 90.99% | 93.15% | 2,886 | 344 |

<sub>*F1-Score calculated as: 2 × (Precision × Recall) / (Precision + Recall)</sub>

#### Detailed Confusion Matrices

**Spam Detection**
```
Confusion Matrix (rows: true, cols: predicted)
                 Predicted
                 POS   NEG
Actual   POS    732     2
         NEG      0  2152
```
- **Precision**: 100.00% (732/732)
- **Recall**: 99.73% (732/734)
- **Performance**: Excellent - near-perfect classification

**Ad Detection**
```
Confusion Matrix (rows: true, cols: predicted)
                 Predicted
                 POS   NEG
Actual   POS    251     4
         NEG      1  2630
```
- **Precision**: 99.60% (251/252)
- **Recall**: 98.43% (251/255)
- **Performance**: Excellent - very high accuracy

**Irrelevant Content Detection**
```
Confusion Matrix (rows: true, cols: predicted)
                 Predicted
                 POS   NEG
Actual   POS    210    13
         NEG      1  2662
```
- **Precision**: 99.53% (210/211)
- **Recall**: 94.17% (210/223)
- **Performance**: Very good - slightly lower recall

**Rant Detection**
```
Confusion Matrix (rows: true, cols: predicted)
                 Predicted
                 POS   NEG
Actual   POS     42     7
         NEG      5  2832
```
- **Precision**: 89.36% (42/47)
- **Recall**: 85.71% (42/49)
- **Performance**: Good - lowest performance due to small sample size

**Unsafe Content Detection**
```
Confusion Matrix (rows: true, cols: predicted)
                 Predicted
                 POS   NEG
Actual   POS    313    31
         NEG     15  2527
```
- **Precision**: 95.43% (313/328)
- **Recall**: 90.99% (313/344)
- **Performance**: Very good - balanced precision and recall

#### Technical Notes

- **Evaluation Method**: Standard binary classification metrics on held-out test set
- **Positive Label**: `__label__pos` (represents harmful/unwanted content)
- **Negative Label**: `__label__neg` (represents acceptable content)
- **Test Set Size**: 2,886 samples per category
- **Model Type**: FastText with character n-grams and word embeddings
