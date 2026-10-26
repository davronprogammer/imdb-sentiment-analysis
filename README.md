# IMDb Sentiment Classification - LSTM vs. GRU

An end-to-end, reproducible deep learning project for binary sentiment classification using PyTorch, comparing **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** architectures on the IMDb Movie Reviews dataset. Includes a static, responsive educational research website with an interactive browser-based sentiment analyzer.
https://imdbsentimental.netlify.app/
---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Problem Statement & Dataset Description](#2-problem-statement--dataset-description)
- [3. Model Architecture Comparison (LSTM vs. GRU)](#3-model-architecture-comparison-lstm-vs-gru)
- [4. Experimental Results & Verification Status](#4-experimental-results--verification-status)
- [5. Audit Findings: Evaluation Threshold Analysis](#5-audit-findings-evaluation-threshold-analysis)
- [6. Repository Structure](#6-repository-structure)
- [7. Installation & Environment Setup](#7-installation--environment-setup)
- [8. Usage Guide (Train, Evaluate, Predict)](#8-usage-guide-train-evaluate-predict)
- [9. Automated Testing](#9-automated-testing)
- [10. Web Application & Interactive Demo](#10-web-application--interactive-demo)
- [11. Netlify Deployment Guide](#11-netlify-deployment-guide)
- [12. Limitations & Project Roadmap](#12-limitations--project-roadmap)
- [13. Contributing Guidelines & Development Workflow](#13-contributing-guidelines--development-workflow)
- [14. License & Credits](#14-license--credits)

---

## 1. Project Overview

This repository explores the comparative behavior of recurrent neural network architectures when classifying English movie reviews as positive or negative. The project emphasizes:
- **Reproducible ML Engineering**: Clean separation of data loading, vocabulary construction, encoding, model definitions, training loops, evaluation, and plotting.
- **Architectural Comparison**: Direct head-to-head analysis between LSTM (3 gates + candidate state) and GRU (2 gates).
- **Leakage Prevention**: Deriving vocabulary exclusively from training data.
- **Audited Metrics**: Correction of an unnormalized logit threshold anomaly discovered in early Colab exploration cells.
- **Educational Web Showcase**: A static, modern, publication-grade research frontend ready for one-click deployment on Netlify.

---

## 2. Problem Statement & Dataset Description

### Problem Formulation
Given a raw, variable-length English movie review $X$, predict the binary sentiment label $Y \in \{0, 1\}$ (where $0$ is negative and $1$ is positive) by computing the posterior probability:

$$P(Y = 1 \mid X) = \sigma(\text{Model}(X))$$

The challenge lies in capturing long-range context and semantic nuance across sequences of up to 200 words without relying on large pretrained transformer backbones.

### Dataset Schema & Splitting
- **Source**: [Stanford AI / Kaggle IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Local File**: `data/IMDB Dataset.csv`
- **Total Rows**: Exactly `50,000` reviews (25,000 positive, 25,000 negative; 0 missing values).
- **Deduplication**: 418 raw duplicates were detected; after text normalization (HTML strip, lowercase, punctuation removal, whitespace normalization), an additional 6 duplicates were resolved. The deduplicated dataset contains **49,576 unique reviews**.
- **Data Splits (Stratified, Seed=42)**:
  - **Train (80%)**: 39,660 reviews
  - **Validation (10%)**: 4,958 reviews
  - **Test (10%)**: 4,958 reviews
- **Vocabulary**: 30,002 tokens (30,000 top words from the training set + `<PAD>` index 0 + `<UNK>` index 1). Maximum sequence length is fixed at 200 tokens with post-padding and truncation.

---

## 3. Model Architecture Comparison (LSTM vs. GRU)

Both models share an identical embedding layer ($V = 30{,}002$, $D = 128$) and a linear classification head ($H = 128 \to 1$). The recurring hidden units represent the primary architectural difference:

### LSTM (Long Short-Term Memory)
Controls memory using three continuous multiplicative gates and a separate internal cell state:
- **Forget Gate**: $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$
- **Input Gate**: $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$
- **Candidate State**: $\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$
- **Cell State Update**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
- **Output Gate**: $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$
- **Hidden State**: $h_t = o_t \odot \tanh(c_t)$

### GRU (Gated Recurrent Unit)
Merges cell and hidden states into a single state $h_t$, governed by two gates:
- **Update Gate**: $z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$
- **Reset Gate**: $r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$
- **Candidate Hidden State**: $\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$
- **Hidden State Update**: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

### Parameter Footprint Breakdown

| Component | Calculation Formula | LSTM (Hidden=128) | GRU (Hidden=128) |
|---|---|---|---|
| **Embedding Layer** | $V \times D = 30{,}002 \times 128$ | 3,840,256 | 3,840,256 |
| **Recurrent Layer** | $K \times (D \times H + H^2 + 2H)$ ($K_{LSTM}=4, K_{GRU}=3$) | 131,840 | 98,816 |
| **Linear Head** | $H \times 1 + 1 = 128 \times 1 + 1$ | 129 | 129 |
| **Total Parameters** | Sum of above | **3,972,225** | **3,939,201** |

*The GRU architecture requires **33,024 fewer parameters** (-25.05% recurrent footprint) while retaining comparable representational capacity.*

---

## 4. Experimental Results & Verification Status

### Environment & Verification Transparency
- **Colab GPU Benchmark (Historical Baseline)**: Conducted on Google Colab using an NVIDIA Tesla T4 GPU over 5 full epochs on 39,660 training samples.
- **Local Diagnostic Status**: Verified locally on CPU using pytest unit tests and diagnostic validation scripts.

### Model Performance Comparison

| Metric | LSTM (Colab GPU) | GRU (Colab GPU) | Advantage |
|---|---|---|---|
| **Test Accuracy** | **86.37%** | 85.48% | **LSTM (+0.89%)** |
| **Positive Class F1-Score** | **86.06%** | 84.55% | **LSTM (+1.51%)** |
| **Macro F1-Score** | **86.36%** | 85.43% | **LSTM (+0.93%)** |
| **Negative Precision** | **84.53%** | 81.41% | **LSTM (+3.12%)** |
| **Negative Recall** | 88.91% | **91.82%** | **GRU (+2.91%)** |
| **Positive Precision** | 88.39% | **90.70%** | **GRU (+2.31%)** |
| **Positive Recall** | **83.84%** | 79.18% | **LSTM (+4.66%)** |
| **Training Duration (5 Epochs)** | 41.17 s | **34.31 s** | **GRU (16.7% faster)** |
| **Inference Latency (Test Set)** | 0.57 s | 0.57 s | Parity |
| **Trainable Parameters** | 3,972,225 | **3,939,201** | **GRU (-33,024 params)** |

---

## 5. Audit Findings: Evaluation Threshold Analysis

During our code review of the initial Google Colab notebook (`notebooks/sentiment_classification.ipynb`), a discrepancy in decision thresholds was uncovered:
1. **Training loop threshold**: During training, accuracy was tracked using `(logits >= 0).float()`. Since the sigmoid function satisfies $\sigma(0) = 0.5$, this corresponds strictly to the standard 50% probability threshold.
2. **Evaluation cells threshold**: Cells 54 & 55 computed predictions with `(outputs >= 0.5).long()`. Because `outputs` were raw, unbounded logits from the linear layer, requiring $\text{logit} \ge 0.5$ placed the probability threshold at:
   $$\sigma(0.5) = \frac{1}{1 + e^{-0.5}} \approx 0.6225$$
3. **Impact**: Requiring 62.25% confidence for a positive prediction artificially deflated positive recall (79.18% on GRU) and inflated negative recall (91.82%).
4. **Resolution**: In `src/evaluation/evaluator.py`, thresholds are consistently configured with `threshold = 0.0` for logits (or `0.5` after applying sigmoid), guaranteeing symmetric, calibrated classification metrics across all evaluation scripts.

---

## 6. Repository Structure

```text
imdb-sentiment-analysis/
├── .gitignore                     # Git hygiene configuration
├── netlify.toml                   # Netlify zero-configuration deployment config
├── pytest.ini                     # Pytest suite configuration
├── requirements.txt               # Pinned Python package dependencies
├── train.py                       # CLI script: end-to-end training and evaluation
├── evaluate.py                    # CLI script: checkpoint evaluation and metric calculation
├── predict.py                     # CLI script: single review inference and interactive shell
├── artifacts/                     # Serialized vocabulary and model checkpoints
│   ├── vocab.json                 # Exported vocabulary token dictionary (30,002 tokens)
│   ├── lstm_best.pt               # Local PyTorch checkpoint for LSTM (git-ignored)
│   └── gru_best.pt                # Local PyTorch checkpoint for GRU (git-ignored)
├── data/                          # Dataset root
│   ├── README.md                  # Dataset acquisition and checksum guide
│   └── IMDB Dataset.csv           # Raw 50,000 movie reviews CSV (git-ignored)
├── docs/                          # In-depth technical documentation
│   ├── PROJECT_PLAN.md            # Architectural blueprint and milestones
│   ├── PROJECT_AUDIT.md           # Repository audit and threshold findings
│   ├── DATASET_AND_PREPROCESSING.md # Cleaning regex, splitting, and tokenization
│   ├── MODEL_ARCHITECTURE.md      # Gate equations, parameters, and diagrams
│   ├── EXPERIMENTS_AND_FINDINGS.md# Training dynamics, loss curves, and metrics
│   ├── MODEL_COMPARISON.md        # Head-to-head engineering trade-off analysis
│   └── WEBSITE_CONTENT.md         # Full text copy for the educational website
├── notebooks/                     # Exploratory research notebooks
│   └── sentiment_classification.ipynb # Original Colab experiment notebook
├── results/                       # Generated evaluation outputs and figures
│   ├── model_comparison.csv       # Tabular benchmark metrics
│   ├── model_comparison.json      # Structured JSON metrics
│   ├── training_history.csv       # Epoch-by-epoch loss and accuracy metrics
│   ├── evaluation_results.json    # Evaluator metrics summary
│   ├── classification_report_lstm.txt # Detailed precision/recall report for LSTM
│   ├── classification_report_gru.txt  # Detailed precision/recall report for GRU
│   ├── training_history.png       # Loss and accuracy progression curve plot
│   ├── confusion_matrices.png     # Side-by-side confusion matrix plot
│   ├── confusion_matrix_lstm.png  # High-resolution LSTM confusion matrix
│   └── confusion_matrix_gru.png   # High-resolution GRU confusion matrix
├── scripts/                       # Utility scripts
│   └── run_diagnostic.py          # Quick CPU verification runner
├── src/                           # Core modular Python package
│   ├── data/                      # Dataset loader, deduplication, and PyTorch DataLoaders
│   ├── features/                  # Regex cleaner, tokenizer, and sequence padding
│   ├── models/                    # SentimentLSTM and SentimentGRU implementations
│   ├── training/                  # Trainer, early stopping, and loss trackers
│   ├── evaluation/                # Evaluator, confusion matrix, and reporting
│   └── utils/                     # Seed management, logging, and I/O handlers
├── tests/                         # Pytest unit and integration test suite
│   └── test_pipeline.py           # Comprehensive end-to-end pipeline tests
└── web/                           # Standalone static web application
    ├── index.html                 # Editorial research report and interactive demo
    ├── README.md                  # Web application documentation and Netlify notes
    ├── css/
    │   └── styles.css             # Publication-grade typography and responsive design
    ├── js/
    │   ├── main.js                # Navigation, smooth scroll, tab controls, modal
    │   └── demo.js                # Browser-based lexicon sentiment analyzer
    └── assets/
        └── images/                # Five custom SVG/PNG editorial illustrations
            ├── imdb-dataset-editorial.png
            ├── editorial-film-bg.png
            ├── data-preparation.png
            ├── recurrent-networks-study.png
            └── sentiment-experiment-lab.png
```

---

## 7. Installation & Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/davronprogammer/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Create and activate a virtual environment

Using `uv` (recommended):
```bash
uv venv --python 3.12
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate
```

Or using standard `venv`:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 8. Usage Guide (Train, Evaluate, Predict)

### End-to-End Training
To train both LSTM and GRU models across 5 epochs on the full dataset:
```bash
python train.py --models both --epochs 5 --batch-size 64
```

To run a fast diagnostic train run on 500 samples (CPU friendly):
```bash
python train.py --max-samples 500 --epochs 2 --batch-size 32
```

### Evaluation
To evaluate a trained checkpoint against the stratified test split:
```bash
# Evaluate LSTM checkpoint
python evaluate.py --model-type lstm --checkpoint artifacts/lstm_best.pt

# Evaluate GRU checkpoint
python evaluate.py --model-type gru --checkpoint artifacts/gru_best.pt
```

### Inference CLI
Run single-sentence inference:
```bash
python predict.py --model-type lstm --text "This movie was an absolute masterpiece with phenomenal acting!"
```

Or launch the interactive terminal prompt:
```bash
python predict.py --model-type lstm
```

---

## 9. Automated Testing

The repository includes a comprehensive `pytest` test suite that verifies data preprocessing, regex normalization, tokenization, sequence encoding, padding, model forward passes, parameter calculations, and evaluation metrics:

```bash
pytest -v tests/
```

All 9 unit tests run within seconds on standard CPUs without requiring an external GPU or preloaded model weights.

---

## 10. Web Application & Interactive Demo

The `web/` directory contains a publication-grade, responsive website designed as an editorial research paper.

### Key Features
- **Editorial Design**: Custom typography, dark-accented palette, and five bespoke technical illustrations.
- **Hero Section**: Responsive header utilizing `web/assets/images/imdb-dataset-editorial.png` as a full-bleed backdrop with WCAG-compliant high-contrast overlay.
- **Interactive Sentiment Demo**: A lightweight, client-side sentiment analyzer in `web/js/demo.js` with instant feedback and sample review chips.

### Important: In-Browser Demo Architecture
> **Note on Demo Inference**: The interactive demo running in the browser uses a fast, rule-based lexicon baseline with negation handling, punctuation stripping, and valence weighting. It is designed to demonstrate classification mechanics on client devices with zero latency. It is **not** running live PyTorch or neural network inference in the browser. For deep neural network predictions using trained weights, use the Python CLI (`python predict.py`).

---

## 11. Netlify Deployment Guide

The web application is 100% static (pure HTML5, CSS3, and modern vanilla JavaScript) with zero backend, server, or database requirements.

### Method A: Git-Integrated Deployment (Recommended)
1. Push this repository to your GitHub account:
   ```bash
   git push origin main
   ```
2. Log into [Netlify](https://app.netlify.com/) and click **"Add new site" > "Import an existing project"**.
3. Select your GitHub repository `davronprogammer/imdb-sentiment-analysis`.
4. Netlify will automatically detect `netlify.toml`:
   - **Base directory**: *(leave empty or set to root)*
   - **Publish directory**: `web`
   - **Build command**: *(leave blank)*
5. Click **"Deploy site"**. Your site will be live on a global CDN in under 15 seconds.

### Method B: Netlify Drop (Manual Drag-and-Drop)
1. Navigate to [Netlify Drop](https://app.netlify.com/drop).
2. Drag and drop the `web` folder directly from your local file explorer into the upload box.
3. Your site will be deployed immediately with an HTTPS URL.

---

## 12. Limitations & Project Roadmap

### Current Limitations
1. **Unpretrained Embeddings**: The embedding layer is trained from scratch; incorporating pretrained vectors (GloVe, FastText) would accelerate early convergence.
2. **Unidirectional Context**: Both models process text strictly left-to-right; bidirectional processing (BiLSTM / BiGRU) would capture right-to-left syntactic context.
3. **Regularization**: Baseline models do not incorporate recurrent dropout, resulting in validation loss plateauing around epoch 4.

### Future Roadmap
- [ ] Export PyTorch checkpoints (`.pt`) to ONNX format.
- [ ] Integrate `onnxruntime-web` into `web/js/` to run actual 128-dimensional neural inference in the browser via WebAssembly / WebGL.
- [ ] Benchmark Bidirectional LSTM (BiLSTM) with attention pooling against the current recurrent baselines.
- [ ] Add Subword / WordPiece tokenization to eliminate out-of-vocabulary (`<UNK>`) occurrences.

---

## 13. Contributing Guidelines & Development Workflow

Contributions, suggestions, and issue reports are welcome.

### Development Workflow
1. **Fork & Branch**: Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Setup Environment**: Install dependencies in a clean virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
3. **Format & Verify**:
   - Verify code passes all unit tests:
     ```bash
     pytest -v tests/
     ```
   - Verify JavaScript syntax:
     ```bash
     node --check web/js/*.js
     ```
4. **Submit PR**: Open a pull request against `main` with a clear explanation of changes.

---

## 14. License & Credits

### License
This project is open-source and available under the [MIT License](LICENSE).

### Credits & Acknowledgments
- **Dataset**: Maas, Andrew L., et al. *"Learning Word Vectors for Sentiment Analysis."* Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
- **Core Libraries**: Built using [PyTorch](https://pytorch.org/), [Scikit-learn](https://scikit-learn.org/), [NumPy](https://numpy.org/), and [Pandas](https://pandas.pydata.org/).
- **Editorial Graphics**: Designed for educational demonstration and scientific publication clarity.
