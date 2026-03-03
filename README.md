# BERT Emotion Classification (SMILE Twitter Dataset)

A simple PyTorch + Hugging Face project that fine-tunes **BERT-base-uncased** to classify tweets into 5 emotions:  
**happy • angry • surprise • sad • disgust**

Built in a single Jupyter Notebook (`Emotion_classification_pytorch_BERT.ipynb`) with early stopping (on PR-AUC), class-weighted loss, and full evaluation metrics.

## What it does
- Loads and cleans the SMILE Twitter emotion dataset
- Tokenizes text with BERT tokenizer
- Trains with proper class balancing and gradient scheduling
- Uses early stopping on validation PR-AUC
- Evaluates on test set with accuracy, precision, recall, F1, AUROC, PR-AUC, etc.

## Tech Stack
- PyTorch + Hugging Face Transformers
- TorchMetrics
- Google Colab (GPU-ready)

## Results
Early stopping triggered at epoch 7 (best val PR-AUC: 0.8940).
Test set (final):

- Accuracy: 0.9606
- AUROC: 0.9907
- Macro F1: 0.5886
- PR-AUC: 0.7839
- Macro Precision: 0.5583
- Macro Recall: 0.6633
---

Made for learning purposes.
