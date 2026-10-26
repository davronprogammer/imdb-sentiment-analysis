# Experiments and Findings — IMDB Sentiment Classification

This document provides a detailed analysis of the experimental setup, training dynamics, overfitting behavior, evaluation results, and key architectural findings from both historical Google Colab runs and local verification.

---

## 1. Experimental Setup & Hyperparameters

Both architectures were trained under identical conditions:

| Hyperparameter | Value | Description |
|---|---|---|
| **Dataset** | IMDb 50k Reviews (49,576 deduplicated) | Binary positive/negative sentiment |
| **Data Split** | 80% Train (39,660) / 10% Val (4,958) / 10% Test (4,958) | Stratified by class label (`seed=42`) |
| **Vocabulary Size** | 30,000 tokens (including `<PAD>=0`, `<UNK>=1`) | Derived strictly from training set |
| **Max Sequence Length** | 200 tokens | Truncated or post-padded |
| **Embedding Dimension** | 128 | Trainable PyTorch `nn.Embedding` |
| **Hidden Dimension** | 128 | Single recurrent layer |
| **Batch Size** | 64 | 620 batches per epoch |
| **Optimizer** | Adam (`lr=0.001`, betas=(0.9, 0.999)) | Standard adaptive moment estimation |
| **Loss Function** | `nn.BCEWithLogitsLoss()` | Binary Cross-Entropy on unnormalized logits |
| **Epochs** | 5 | Determined after 10-epoch exploratory run |
| **Model Selection** | Lowest Validation Loss Checkpoint | Restored for final test evaluation |

---

## 2. Training and Validation Dynamics

### Historical Colab 5-Epoch Progression

#### LSTM Training History (Historical GPU Run)
- **Epoch 1**: Train Loss: `0.6921` | Val Loss: `0.6956` | Train Acc: `51.69%` | Val Acc: `50.46%`
- **Epoch 2**: Train Loss: `0.6345` | Val Loss: `0.4947` | Train Acc: `62.66%` | Val Acc: `78.58%`
- **Epoch 3**: Train Loss: `0.3621` | Val Loss: `0.3375` | Train Acc: `84.89%` | Val Acc: `85.22%`
- **Epoch 4**: Train Loss: `0.2400` | Val Loss: `0.3187` | Train Acc: `90.73%` | Val Acc: `86.97%` [BEST VAL LOSS]
- **Epoch 5**: Train Loss: `0.1649` | Val Loss: `0.3451` | Train Acc: `94.20%` | Val Acc: `87.47%`
- **Total Training Duration**: `41.17 seconds` (Colab NVIDIA T4 GPU)

#### GRU Training History (Historical GPU Run)
- **Epoch 1**: Train Loss: `0.6941` | Val Loss: `0.6907` | Train Acc: `50.90%` | Val Acc: `57.58%`
- **Epoch 2**: Train Loss: `0.5998` | Val Loss: `0.5627` | Train Acc: `65.15%` | Val Acc: `66.68%`
- **Epoch 3**: Train Loss: `0.4619` | Val Loss: `0.4452` | Train Acc: `75.86%` | Val Acc: `75.68%`
- **Epoch 4**: Train Loss: `0.3184` | Val Loss: `0.3435` | Train Acc: `86.33%` | Val Acc: `85.11%`
- **Epoch 5**: Train Loss: `0.2199` | Val Loss: `0.3202` | Train Acc: `91.55%` | Val Acc: `86.65%` [BEST VAL LOSS]
- **Total Training Duration**: `34.31 seconds` (Colab NVIDIA T4 GPU)

### Computational Speed Observation
On the GPU benchmark, GRU trained in **34.31 seconds** versus LSTM's **41.17 seconds**—an execution speedup of approximately **16.7%**. This directly aligns with GRU's simpler 2-gate architecture requiring 25% fewer recurrent matrix operations.

---

## 3. Overfitting Analysis & Optimal Stopping

In an initial 10-epoch exploratory experiment documented in the notebook:
- Training loss continued descending steeply toward ~0.05, and training accuracy surpassed 98%.
- However, validation loss hit a plateau around Epoch 4 (LSTM: `0.3187`) and Epoch 5 (GRU: `0.3202`), after which validation loss reversed and began climbing upward toward `0.45 - 0.55`.
- This classic divergence indicates that without regularization (such as dropout or weight decay), 5 epochs is the empirical saturation threshold for simple RNNs on this dataset.
- Selecting the checkpoint with the lowest validation loss successfully prevented overfitted weights from being evaluated on the test set.

---

## 4. Critical Audit Finding: The Logit Decision Threshold Bug

During our deep audit of `notebooks/sentiment_classification.ipynb`, an important inconsistency was identified in the historical Colab evaluation code:

1. **Training loop (Cell 50)**:
   ```python
   predictions = (logits >= 0).float()
   ```
   This is mathematically accurate. In PyTorch `BCEWithLogitsLoss`, the model emits raw logits $z \in (-\infty, +\infty)$. The standard decision boundary for binary classification is $\sigma(z) \ge 0.5$, which simplifies directly to $z \ge 0.0$.

2. **Evaluation function (Cell 54 & Cell 55)**:
   ```python
   predictions = (outputs >= 0.5).long()
   ```
   Here, `outputs` was the raw linear head logit $z$, **not** the sigmoid probability!
   Comparing a raw logit to $\ge 0.5$ shifts the decision threshold from $P(\text{positive}) \ge 0.50$ to $P(\text{positive}) \ge \sigma(0.5) \approx 0.6225$.

### Effect on Historical Metrics
This threshold shift explains why the historical Colab metrics exhibited an artificial asymmetry:
- **Negative Recall was artificially elevated** (GRU: 91.82%, LSTM: 88.91%) because any review with an uncertain positive prediction ($0.0 \le \text{logit} < 0.5$) was classified as negative.
- **Positive Recall was suppressed** (GRU: 79.18%, LSTM: 83.84%).
- **Positive Precision was artificially inflated** (GRU: 90.70%, LSTM: 88.39%) because only high-confidence positive reviews ($\text{logit} \ge 0.5$) were classified as positive.

In our refactored pipeline (`src/evaluation/evaluator.py`), this has been made explicit: the default threshold is mathematically correct at `threshold = 0.0` (or `torch.sigmoid(logits) >= 0.5`), with clear documentation of the historical behavior.

---

## 5. Comparative Evaluation Results

### Historical Colab Test Set Results (4,958 reviews)

| Metric | LSTM (Historical Colab) | GRU (Historical Colab) | Winner |
|---|---|---|---|
| **Test Accuracy** | **86.37%** (0.8637) | 85.48% (0.8548) | **LSTM (+0.89%)** |
| **Positive F1-Score** | **86.06%** (0.8606) | 84.55% (0.8455) | **LSTM (+1.51%)** |
| **Macro F1-Score** | **86.36%** (0.8636) | 85.43% (0.8543) | **LSTM (+0.93%)** |
| **Negative Precision** | **84.53%** (0.8453) | 81.41% (0.8141) | **LSTM (+3.12%)** |
| **Negative Recall** | 88.91% (0.8891) | **91.82%** (0.9182) | **GRU (+2.91%)** |
| **Positive Precision** | 88.39% (0.8839) | **90.70%** (0.9070) | **GRU (+2.31%)** |
| **Positive Recall** | **83.84%** (0.8384) | 79.18% (0.7918) | **LSTM (+4.66%)** |
| **Evaluation Time** | 0.57 s | 0.57 s | Tie |
| **Training Time** | 41.17 s | **34.31 s** | **GRU (16.7% faster)** |
| **Trainable Parameters** | 3,972,225 | **3,939,201** | **GRU (33,024 fewer)** |

---

## 6. Confusion Matrix Interpretation (Historical 4,958 Test Samples)

### LSTM Confusion Matrix
- **True Negatives (TN)**: 2,196 / 2,470 (88.91% specificity)
- **False Positives (FP)**: 274 / 2,470 (11.09%)
- **False Negatives (FN)**: 402 / 2,488 (16.16%)
- **True Positives (TP)**: 2,086 / 2,488 (83.84% sensitivity)

### GRU Confusion Matrix
- **True Negatives (TN)**: 2,268 / 2,470 (91.82% specificity)
- **False Positives (FP)**: 202 / 2,470 (8.18%)
- **False Negatives (FN)**: 518 / 2,488 (20.82%)
- **True Positives (TP)**: 1,970 / 2,488 (79.18% sensitivity)

### Diagnostic Comparison
- LSTM made **676 total errors** (274 FP + 402 FN).
- GRU made **720 total errors** (202 FP + 518 FN).
- LSTM achieved a more balanced distribution between false positive and false negative errors, making it superior for general sentiment classification where neither class error is preferred.

---

## 7. Limitations & Recommended Improvements

1. **Pretrained Word Representations**:
   The current models train 128-dimensional embeddings from scratch on the 39,660 training reviews. Initializing embeddings with pretrained vectors (e.g. GloVe 100d/300d or Word2Vec) typically boosts accuracy to 88-90% by providing semantic grounding for rare words.
2. **Regularization**:
   Neither model employs dropout or weight decay. Adding `nn.Dropout(0.3)` after the embedding layer and before the classification head would mitigate the validation loss divergence observed after 4-5 epochs.
3. **Bidirectional Recurrence**:
   Switching from unidirectional to bidirectional recurrence (`bidirectional=True`) would enable the recurrent layers to process context from both forward and backward directions, which is particularly beneficial for sentiment where qualifiers appear late in sentences.
4. **Packed Sequences**:
   Currently, sequences are statically padded to length 200. Using PyTorch `pack_padded_sequence` with dynamic batch lengths reduces wasted compute on padding tokens and avoids passing padded states through the final linear projection.
