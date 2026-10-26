# Project Audit — IMDB Sentiment Analysis

**Date**: October 2026  
**Auditor**: Machine Learning Engineering & Maintainer Team  
**Project**: IMDB Sentiment Classification — LSTM vs. GRU

---

## 1. Executive Summary

This audit assesses the state of the `imdb-sentiment-analysis` repository, reviewing code quality, architectural consistency, dataset integrity, experimental records from Google Colab, and readiness for a professional GitHub portfolio and frontend showcase.

The codebase successfully transitions an educational single-notebook prototype (`notebooks/sentiment_classification.ipynb`) into a production-grade, modular Python package (`src/`) with reproducible command-line interfaces, comprehensive unit tests, and thorough documentation.

---

## 2. Directory Structure and Asset Classification

| Path | Asset Type | Status | Description |
|---|---|---|---|
| `data/IMDB Dataset.csv` | Raw Data | Present (66.2 MB) | 50,000 raw movie reviews with binary sentiment labels. Excluded from Git to prevent repository bloat. |
| `data/README.md` | Documentation | Tracked | Dataset provenance, citation, and Kaggle download instructions. |
| `notebooks/sentiment_classification.ipynb` | Original Prototype | Tracked | Complete 57-cell educational notebook from Google Colab. |
| `src/data/` | Source Code | Tracked | Data loading, deduplication, label mapping, and PyTorch DataLoaders. |
| `src/features/` | Source Code | Tracked | Text cleaning, tokenization, vocabulary management, and sequence encoding. |
| `src/models/` | Source Code | Tracked | PyTorch `SentimentLSTM` and `SentimentGRU` implementations. |
| `src/training/` | Source Code | Tracked | PyTorch `Trainer` with validation tracking and early stopping. |
| `src/evaluation/` | Source Code | Tracked | Metrics computation, prediction generation, and Matplotlib plotting. |
| `src/utils/` | Source Code | Tracked | Configuration dataclass, random seed helper, and checkpoint I/O. |
| `tests/` | Test Suite | Tracked | 9 automated pytest unit tests covering preprocessing, tokenization, model shapes, and metrics. |
| `artifacts/vocab.json` | Model Artifact | Tracked (223 KB) | Serialized top-30,000 vocabulary mapping derived strictly from training data. |
| `artifacts/*.pt` | Model Artifacts | Ignored (~5.7 MB) | Best model weights for LSTM and GRU. Generated via `train.py`. |
| `results/` | Reports & Figures | Tracked | Comparison CSV/JSON, training curves, confusion matrices, and classification reports. |
| `docs/` | Technical Docs | Tracked | In-depth technical guides, architecture specifications, and website copy. |
| `web/README.md` | Placeholder | Tracked | Future static frontend roadmap. |

---

## 3. Implemented Functionality

1. **Text Normalization**:
   - Sequential cleaning pipeline: lowercase conversion, HTML entity unescaping, `<br>` tag removal, full HTML tag stripping, URL elimination, numeric digit removal, punctuation stripping, and whitespace normalization.
2. **Deduplication**:
   - Detected 418 raw duplicates; an additional 6 duplicates were resolved following text normalization, yielding 49,576 unique reviews.
3. **Stratified Splitting**:
   - Clean, reproducible two-stage split with `random_state=42`: 80% train (39,660), 10% validation (4,958), and 10% test (4,958).
4. **Leakage-Free Vocabulary**:
   - `Vocabulary` built exclusively on the training split, reserving index 0 for `<PAD>` and index 1 for `<UNK>`.
5. **Sequence Processing**:
   - Reviews truncated or post-padded to 200 tokens.
6. **Model Architectures**:
   - `SentimentLSTM`: 128-dim embedding, 128-dim hidden state, linear head (3,972,225 parameters).
   - `SentimentGRU`: 128-dim embedding, 128-dim hidden state, linear head (3,939,201 parameters).
7. **Training Engine**:
   - `Trainer` with Adam optimizer (`lr=0.001`), `nn.BCEWithLogitsLoss`, validation checkpointing, and configurable early stopping.
8. **Command-Line Interfaces**:
   - `train.py`: Full training, evaluation, and plot generation pipeline.
   - `evaluate.py`: Standalone checkpoint evaluation on test split.
   - `predict.py`: Real-time inference on arbitrary input text strings.

---

## 4. Current ML Results

### Historical Benchmark (Google Colab, NVIDIA T4 GPU, 4,958 Test Reviews)
- **LSTM**:
  - Test Accuracy: `86.37%`
  - Positive-class F1-score: `86.06%`
  - Macro F1-score: `86.36%`
  - Negative Recall: `88.91%`
  - Positive Recall: `83.84%`
  - Trainable parameters: `3,972,225`
  - Training duration: `41.17 seconds`
  - Test evaluation duration: `0.57 seconds`
- **GRU**:
  - Test Accuracy: `85.48%`
  - Positive-class F1-score: `84.55%`
  - Macro F1-score: `85.43%`
  - Negative Recall: `91.82%`
  - Positive Recall: `79.18%`
  - Trainable parameters: `3,939,201`
  - Training duration: `34.31 seconds`
  - Test evaluation duration: `0.57 seconds`

### Local Diagnostic Status
- Executed on local CPU with `--max-samples 500 --epochs 2` to verify checkpoint creation, metrics serialization, and plot rendering without requiring hours of CPU compute.
- All 9 automated tests passed with 100% pass rate.

---

## 5. Known Issues & Audit Discoveries

1. **Logit Evaluation Threshold Discrepancy**:
   - *Discovery*: In the original Colab notebook (Cells 54 & 55), test evaluation checked `(outputs >= 0.5).long()` where `outputs` was raw logits from `nn.Linear(128, 1)`.
   - *Consequence*: This shifted the decision threshold from probability $0.5$ ($\text{logit} = 0.0$) to probability $\approx 0.6225$ ($\text{logit} = 0.5$), skewing positive recall downward (79.18% on GRU) and negative recall upward (91.82% on GRU).
   - *Resolution*: Corrected in `src/evaluation/evaluator.py` where threshold defaults to `0.0` (matching probability 0.5).
2. **Missing Regularization**:
   - Baseline models omit dropout (`nn.Dropout`), resulting in validation loss plateauing around Epoch 4-5.
3. **Static Padding**:
   - Reviews are statically padded to length 200 rather than dynamically packed via `torch.nn.utils.rnn.pack_padded_sequence`.

---

## 6. Missing Components (Addressed in this Phase)

- **Git Version Control**: Repository was previously uninitialized. Initialized Git with conventional commit milestones and established `.gitignore`.
- **Dataset Policy**: Formalized that `data/IMDB Dataset.csv` (66.2 MB) remains untracked in Git to respect repository size limits, while documenting provenance in `data/README.md`.
- **Model Checkpoint Policy**: Untracked heavy binary `.pt` files in `.gitignore` while tracking `artifacts/vocab.json`.

---

## 7. Recommended Next Steps

1. **Commit History**: Finalize conventional commits tracking modularization, testing, and documentation milestones.
2. **Phase 2 — Static Frontend Web Showcase**: Build the frontend website in `web/` using HTML5, modern CSS, and vanilla JavaScript, leveraging content from `docs/WEBSITE_CONTENT.md`.
3. **ONNX Browser Runtime**: Export `SentimentLSTM` weights to ONNX format with 8-bit dynamic quantization for real-time in-browser inference via `onnxruntime-web`.
