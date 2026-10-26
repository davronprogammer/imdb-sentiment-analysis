# IMDb Sentiment Classification — LSTM vs. GRU

An end-to-end, reproducible deep learning project for binary sentiment classification using PyTorch, comparing **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** architectures on the IMDb Movie Reviews dataset.

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Dataset Description](#3-dataset-description)
- [4. Technologies Used](#4-technologies-used)
- [5. Project Structure](#5-project-structure)
- [6. Installation & Environment Setup](#6-installation--environment-setup)
- [7. Training & Evaluation Pipeline](#7-training--evaluation-pipeline)
- [8. Experimental Results & Verification Status](#8-experimental-results--verification-status)
- [9. Audit Findings: Threshold Inconsistency](#9-audit-findings-threshold-inconsistency)
- [10. Limitations & Future Roadmap](#10-limitations--future-roadmap)
- [11. Documentation Index](#11-documentation-index)

---

## 1. Project Overview

This repository explores the comparative behavior of recurrent neural network architectures when classifying English movie reviews as positive or negative. The project emphasizes:
- **Reproducible ML Engineering**: Clean separation of data loading, vocabulary construction, encoding, model definitions, training loops, evaluation, and plotting.
- **Architectural Comparison**: Direct head-to-head analysis between LSTM (3 gates + cell state) and GRU (2 gates).
- **Leakage Prevention**: Deriving vocabulary exclusively from training data.
- **Educational Value**: Clear mathematical derivations and factual documentation preparing the project for a future interactive educational website.

---

## 2. Problem Statement

Given a raw, variable-length English movie review $X$, predict the binary sentiment label $Y \in \{0, 1\}$ (where $0$ is negative and $1$ is positive) by computing the posterior probability:

$$P(Y = 1 \mid X) = \sigma(\text{Model}(X))$$

The challenge lies in capturing long-range context and semantic nuance across sequences of up to 200 words without relying on large pretrained transformer backbones.

---

## 3. Dataset Description

- **Source**: [Kaggle IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Local File**: `data/IMDB Dataset.csv`
- **Total Rows**: Exactly `50,000` reviews (25,000 positive, 25,000 negative; 0 missing values).
- **Deduplication**: 418 raw duplicates were detected; after text normalization (cleaning HTML tags, punctuation, and whitespace), an additional 6 duplicates were resolved. The deduplicated dataset contains **49,576 unique reviews**.
- **Data Splits (Stratified, Seed=42)**:
  - **Train (80%)**: 39,660 reviews
  - **Validation (10%)**: 4,958 reviews
  - **Test (10%)**: 4,958 reviews

---

## 4. Technologies Used

- **Language**: Python (>= 3.10, tested on 3.12)
- **Deep Learning**: [PyTorch](https://pytorch.org/) (CPU and CUDA compatible)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Evaluation & Metrics**: [Scikit-learn](https://scikit-learn.org/)
- **Visualizations**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Testing**: [pytest](https://docs.pytest.org/)
- **Package Management**: [uv](https://docs.astral.sh/uv/) / pip

---

## 5. Project Structure

```text
imdb-sentiment-analysis/
├── artifacts/              # Serialized vocabulary and best model checkpoints (.pt)
│   ├── vocab.json
│   ├── lstm_best.pt
│   └── gru_best.pt
├── data/                   # Dataset directory
│   └── IMDB Dataset.csv    # 50,000 movie reviews
├── docs/                   # In-depth technical and educational documentation
│   ├── PROJECT_PLAN.md
│   ├── DATASET_AND_PREPROCESSING.md
│   ├── MODEL_ARCHITECTURE.md
│   ├── EXPERIMENTS_AND_FINDINGS.md
│   ├── MODEL_COMPARISON.md
│   └── WEBSITE_CONTENT.md
├── notebooks/              # Original Colab exploration notebook
│   └── sentiment_classification.ipynb
├── results/                # Evaluation metrics, comparison tables, and figures
│   ├── model_comparison.csv
│   ├── model_comparison.json
│   ├── training_history.csv
│   ├── evaluation_results.json
│   ├── classification_report_lstm.txt
│   ├── classification_report_gru.txt
│   ├── training_history.png
│   ├── confusion_matrices.png
│   ├── confusion_matrix_lstm.png
│   └── confusion_matrix_gru.png
├── src/                    # Modular source package
│   ├── data/               # CSV loader, deduplication, dataset, dataloaders
│   ├── features/           # Regex text cleaner, tokenizer, vocabulary, sequence encoder
│   ├── models/             # SentimentLSTM, SentimentGRU, parameter counters
│   ├── training/           # PyTorch Trainer, early stopping, validation tracking
│   ├── evaluation/         # Metrics computation, evaluation runner, plot generators
│   └── utils/              # Configuration, seeding, and I/O helpers
├── tests/                  # Automated unit test suite
│   └── test_pipeline.py
├── evaluate.py             # CLI script for evaluating saved checkpoints
├── predict.py              # CLI and interactive sentiment inference script
├── pytest.ini              # Pytest configuration
├── requirements.txt        # Dependency specification
├── train.py                # CLI pipeline for training, evaluation, and plot generation
└── README.md
```

---

## 6. Installation & Environment Setup

### 1. Clone repository
```bash
git clone https://github.com/username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Create and activate a virtual environment
Using `uv`:
```bash
uv venv --python 3.12
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux
```

Or using standard `venv`:
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run automated unit tests
```bash
pytest -v tests/
```

---

## 7. Training & Evaluation Pipeline

### Run Full Training Pipeline (LSTM + GRU)
```bash
python train.py --models both --epochs 5 --batch-size 64
```

### Run Fast Diagnostic Verification (e.g. 500 samples on CPU)
```bash
python train.py --max-samples 500 --epochs 2 --batch-size 32
```

### Evaluate Existing Checkpoint
```bash
python evaluate.py --model-type lstm --checkpoint artifacts/lstm_best.pt
```

### Run Sentiment Prediction on New Text
```bash
python predict.py --model-type lstm --text "This movie was an absolute masterpiece with phenomenal acting!"
```
Or launch interactive prompt:
```bash
python predict.py --model-type lstm
```

---

## 8. Experimental Results & Verification Status

### Status Distinction
- **Historical Google Colab Results**: Full 5-epoch training on 39,660 reviews using an NVIDIA T4 GPU.
- **Local Diagnostic Status**: Pipeline verified end-to-end on local CPU with unit test verification and diagnostic sample runs.

### Benchmark Results Comparison (Historical Full Run)

| Metric | LSTM (Colab GPU) | GRU (Colab GPU) | Superior Model |
|---|---|---|---|
| **Test Accuracy** | **86.37%** | 85.48% | **LSTM (+0.89%)** |
| **Positive Class F1-Score** | **86.06%** | 84.55% | **LSTM (+1.51%)** |
| **Macro F1-Score** | **86.36%** | 85.43% | **LSTM (+0.93%)** |
| **Negative Precision** | **84.53%** | 81.41% | **LSTM (+3.12%)** |
| **Negative Recall** | 88.91% | **91.82%** | **GRU (+2.91%)** |
| **Positive Precision** | 88.39% | **90.70%** | **GRU (+2.31%)** |
| **Positive Recall** | **83.84%** | 79.18% | **LSTM (+4.66%)** |
| **Training Duration** | 41.17 s | **34.31 s** | **GRU (16.7% faster)** |
| **Evaluation Latency** | 0.57 s | 0.57 s | Tie |
| **Trainable Parameters** | 3,972,225 | **3,939,201** | **GRU (33,024 fewer)** |

---

## 9. Audit Findings: Threshold Inconsistency

During the audit of the original Colab notebook (`notebooks/sentiment_classification.ipynb`), a critical evaluation inconsistency was discovered:
- In the training loop, predictions were computed using `(logits >= 0).float()`, which corresponds accurately to $\sigma(\text{logit}) \ge 0.5$.
- In the notebook's evaluation cells (54 & 55), predictions were computed using `(outputs >= 0.5).long()`.
- Because `outputs` were raw unnormalized logits from the linear layer, checking `logit >= 0.5` inadvertently shifted the decision boundary to probability $P \ge \sigma(0.5) \approx 0.6225$.
- This explains why historical negative recall was artificially high (91.82% on GRU) and positive recall was reduced.
- In `src/evaluation/evaluator.py`, this threshold has been standardized to `threshold = 0.0` (matching probability 0.5).

---

## 10. Limitations & Future Roadmap

1. **Current Limitations**:
   - Embeddings are trained from scratch without pretrained weights (GloVe or Word2Vec).
   - Recurrent layers are unidirectional.
   - No dropout was included in the baseline architecture, causing validation loss to plateau after 4-5 epochs.
2. **Next Steps for Website Showcase**:
   - Export best LSTM checkpoint to ONNX format.
   - Implement client-side in-browser inference using `onnxruntime-web` or TensorFlow.js.
   - Build a modern, responsive frontend (HTML, CSS, JavaScript) hosted statically on Netlify.

---

## 11. Documentation Index

For comprehensive technical derivations and educational content, refer to the `docs/` directory:
- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md): Objectives, technical decisions, and scope boundaries.
- [docs/DATASET_AND_PREPROCESSING.md](docs/DATASET_AND_PREPROCESSING.md): Detailed dataset schema, deduplication, cleaning regex, and split verification.
- [docs/MODEL_ARCHITECTURE.md](docs/MODEL_ARCHITECTURE.md): Mathematical gate derivations and exact parameter calculations.
- [docs/EXPERIMENTS_AND_FINDINGS.md](docs/EXPERIMENTS_AND_FINDINGS.md): Detailed training dynamics, overfitting analysis, and threshold audit.
- [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md): In-depth trade-off analysis between LSTM and GRU.
- [docs/WEBSITE_CONTENT.md](docs/WEBSITE_CONTENT.md): 17 structured sections of factual copy for the upcoming educational website.
