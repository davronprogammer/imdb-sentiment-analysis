# Model Comparison — LSTM vs. GRU

A comprehensive comparative analysis of Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models evaluated on the IMDb sentiment classification task.

---

## 1. Executive Summary Table

| Evaluation Criterion | LSTM | GRU | Differential ($\Delta$) | Superior Model |
|---|---|---|---|---|
| **Test Accuracy** | **86.37%** | 85.48% | $+0.89\%$ | **LSTM** |
| **Positive F1-Score** | **86.06%** | 84.55% | $+1.51\%$ | **LSTM** |
| **Macro F1-Score** | **86.36%** | 85.43% | $+0.93\%$ | **LSTM** |
| **Negative Precision** | **84.53%** | 81.41% | $+3.12\%$ | **LSTM** |
| **Negative Recall** | 88.91% | **91.82%** | $-2.91\%$ | **GRU** |
| **Positive Precision** | 88.39% | **90.70%** | $-2.31\%$ | **GRU** |
| **Positive Recall** | **83.84%** | 79.18% | $+4.66\%$ | **LSTM** |
| **Trainable Parameters** | 3,972,225 | **3,939,201** | $-33,024$ ($-0.83\%$) | **GRU** |
| **Recurrent Layer Params** | 132,096 | **99,072** | $-33,024$ ($-25.0\%$) | **GRU** |
| **Training Duration (GPU)** | 41.17 s | **34.31 s** | $-6.86\text{ s}$ ($-16.7\%$) | **GRU** |
| **Evaluation Time (GPU)** | 0.57 s | 0.57 s | $0.00\text{ s}$ | **Tie** |
| **Uncompressed Model Size** | ~15.2 MB | **~15.0 MB** | $-0.13\text{ MB}$ | **GRU** |

*Note: The performance metrics, training times, and evaluation times above reflect the recorded historical experiments conducted on Google Colab with NVIDIA GPU acceleration on the full 49,576-review dataset.*

---

## 2. In-Depth Metric Analysis

### Overall Accuracy & F1 Performance
- **LSTM** proved superior in holistic classification accuracy ($86.37\%$ vs $85.48\%$) and Positive-class F1-score ($86.06\%$ vs $84.55\%$).
- The Macro F1-score of LSTM ($86.36\%$) demonstrates a more harmonized trade-off between positive and negative class identification compared to GRU ($85.43\%$).

### Class Sensitivity (Precision vs. Recall Trade-Off)
- **GRU** exhibited an asymmetric affinity toward negative predictions, capturing **91.82%** of all negative reviews, but consequently suffering a lower recall (**79.18%**) on positive reviews.
- **LSTM** maintained balanced sensitivity across both classes (**88.91%** negative recall vs. **83.84%** positive recall).

---

## 3. Computational and Memory Profiles

### Parameter Footprint
- The bulk of parameters in both networks resides in the shared embedding layer ($30{,}000 \times 128 = 3{,}840{,}000$ parameters, representing $96.7\%$ of the entire model).
- Within the recurrent core:
  - LSTM requires 4 internal gate transformations: $4 \times (128^2 + 128^2 + 256) = 132{,}096$ weights.
  - GRU requires 3 internal gate transformations: $3 \times (128^2 + 128^2 + 256) = 99{,}072$ weights.
- As a result, the GRU recurrent module is **25.0% smaller** than LSTM, yielding an overall model parameter reduction of 33,024 weights.

### Compute Throughput
- During full 5-epoch GPU training on Colab, GRU finished in **34.31 seconds** compared to LSTM's **41.17 seconds** (a **16.7% speedup**).
- For batched inference over 4,958 test reviews, both models completed the forward pass in **0.57 seconds** on GPU, indicating that latency in batch inference is dominated by embedding lookups and tensor transfers rather than gate operations.

---

## 4. Strengths & Weaknesses

### Long Short-Term Memory (LSTM)
**Strengths:**
- Higher overall predictive accuracy (+0.89%) and F1-score (+1.51%).
- Better retention of complex linguistic nuance over 200-word sequences, owing to decoupled cell and hidden states.
- Lower total error count (676 vs. 720 errors).
- Balanced false-positive and false-negative profile.

**Weaknesses:**
- Higher computational demand (4 gates per recurrent timestep).
- 16.7% slower training time on equivalent hardware.
- Slightly higher memory and checkpoint footprint (+33k parameters).

---

### Gated Recurrent Unit (GRU)
**Strengths:**
- Faster training convergence and execution (34.3s vs. 41.2s).
- 25% fewer parameters inside the recurrent mechanism.
- Exceptional negative review identification (91.82% negative recall).
- Simpler mathematical graph, making it easier to implement and export to edge environments.

**Weaknesses:**
- Lower overall classification accuracy (85.48%).
- Significant false-negative rate on positive reviews (518 false negatives vs. LSTM's 402).
- Faster susceptibility to overfitting if trained beyond 4-5 epochs without regularization.

---

## 5. Final Recommendation

For this binary sentiment classification task:

> **Recommended Model**: **Long Short-Term Memory (LSTM)**

### Decision Justification:
1. **Primary Evaluation Metric**: In customer review and sentiment classification, general accuracy and balanced F1-score are the primary performance indicators. LSTM outperformed GRU across Accuracy (86.37%), F1 (86.06%), and Macro F1 (86.36%).
2. **Acceptable Resource Trade-Off**: Because the embedding matrix dominates total model size (96.7%), GRU's recurrent parameter savings of 33,024 weights translate to less than a 1% overall file size difference (~15.0 MB vs. ~15.2 MB uncompressed). The marginal compute savings (6.8 seconds during training) do not outweigh LSTM's consistent +0.89% gain in generalization accuracy.
3. **Deployment Strategy**: For the future frontend portfolio website, LSTM serves as the primary showcased classifier, while GRU serves as the ideal educational comparison showing how gate simplification impacts training speed and class recall.
