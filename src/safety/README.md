# Safety Model Classifer Usage Guide

This guide explains how to use the Safety Model classifier for content moderation. The workflow consists of three main steps:

1. **TF-IDF**: Process data using TF-IDF
2. **Lexicon Filter**: Automatically reject reviews that contain unacceptable (and therefore unsafe) language
3. **Logistic Regression**: Analyse text that may have slipped through the lexicon filter to predict if it is safe or not

To reproduce the evaluation metrics for yourself, run this from root: 

```bash
uv run -m src.safety.safety_eval
```

### Performance 

Please see the table below for a summary of the evaluation performance of the Safety Model on a held-out test dataset:

#### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.89      | 0.87   | 0.88     | 66,398  |
| 1     | 0.80      | 0.82   | 0.81     | 40,971  |

**Accuracy:** 0.85 (107,369 samples)  

| Metric        | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Macro avg     | 0.84      | 0.85   | 0.85     | 107,369 |
| Weighted avg  | 0.85      | 0.85   | 0.85     | 107,369 |

### Analysis (A quick one) 

While recall is often considered the priority for identifying toxic text, we confirmed with experimentation that this often came at the expense of recall, causing a lot of normal reviews to get flagged as toxic. We identified this is as a potential annoyance that can drive away users, so we maximise f1 score in our training, with the confidence that the majority of unsafe texts will be filtered out with this model, and those that slip through will be identifying at the subsequent stages of our pipeline. 

The model parameters resulting in the best performance have been saved under `models/`.