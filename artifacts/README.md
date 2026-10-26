# Model Artifacts Directory

This directory stores trained model checkpoints and serialized vocabulary mappings required for deterministic inference.

## Contents
- `vocab.json` (tracked in Git): JSON serialized vocabulary mapping top 30,000 words, including `<PAD>` (index 0) and `<UNK>` (index 1), constructed exclusively from the training split.
- `lstm_best.pt` (ignored from Git): Best-performing weights and training metadata for `SentimentLSTM`.
- `gru_best.pt` (ignored from Git): Best-performing weights and training metadata for `SentimentGRU`.

## Checkpoint Policy
Trained PyTorch weights (`*.pt`) are binary model artifacts (~5.7 MB each). They are generated dynamically via:
```bash
python train.py --models both --epochs 5 --batch-size 64
```
To preserve local reproducibility while keeping Git history lightweight and clean, binary `.pt` files are excluded in `.gitignore`.
