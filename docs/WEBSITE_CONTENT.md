# Educational Website Content — IMDB Sentiment Classification

This document provides structured, factual, and evidence-based educational content designed directly for the future frontend-only showcase website. Every technical stage contains four core educational pillars:
1. **What was done**
2. **Why it was necessary**
3. **What was observed**
4. **The actual result or verified metric**

---

## 1. Project Introduction

- **What was done**: We engineered an end-to-end natural language processing (NLP) project that trains, validates, evaluates, and rigorously compares two recurrent neural network architectures—Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)—for binary sentiment analysis on movie reviews.
- **Why it was necessary**: Recurrent neural networks represent the historical and conceptual foundation of modern sequential deep learning. Directly comparing LSTM and GRU on identical data splits reveals how internal gating mechanisms affect learning dynamics, parameter counts, training latency, and classification trade-offs.
- **What was observed**: Both architectures successfully learned to map natural language sequences to sentiment polarity, achieving over 85% test accuracy from raw text without pretrained language models.
- **Actual result**: Built an educational codebase with a modular pipeline (`src/`), reproducible training scripts, automated unit tests, and verified empirical benchmarks.

---

## 2. Problem and Objective

- **What was done**: Formulated sentiment classification as a supervised binary sequence classification problem, mapping a variable-length English review to a probability score $P(\text{sentiment} = \text{positive} \mid \text{review})$.
- **Why it was necessary**: Unstructured text contains colloquialisms, irony, and complex syntactic dependencies. Binary sentiment classification serves as the gold-standard benchmark for evaluating sequential sequence comprehension.
- **What was observed**: Reviews range from short single-paragraph reactions to lengthy thousand-word essays, requiring standardized truncation and padding.
- **Actual result**: Defined target metrics for model comparison: Accuracy, Precision, Recall, Macro/Weighted F1-score, trainable parameters, training latency, and inference time.

---

## 3. Dataset Overview

- **What was done**: Utilized the standard IMDb Large Movie Review dataset stored at `data/IMDB Dataset.csv`.
- **Why it was necessary**: A recognized, widely benchmarked dataset enables objective performance comparisons against established literature.
- **What was observed**: The dataset consists of 50,000 labeled reviews with an exact 50/50 class balance (25,000 positive, 25,000 negative) and zero missing values in either column.
- **Actual result**:
  - Raw rows: `50,000`
  - Columns: `review` (text), `sentiment` (categorical)
  - Missing values: `0`
  - Raw duplicates: `418`

---

## 4. Data-Cleaning Process

- **What was done**: Developed an 8-stage text normalization pipeline (`clean_text`): lowercasing, HTML entity unescaping (`&amp;` $\to$ `&`), `<br>` tag removal, arbitrary HTML tag stripping, URL removal, digit stripping, non-letter punctuation stripping, and whitespace collapse. Deduplicated the dataset after cleaning.
- **Why it was necessary**: Raw web-scraped reviews contain noisy HTML formatting, links, and erratic punctuation that bloat vocabulary size without contributing sentiment signal.
- **What was observed**: Normalization revealed 6 additional duplicate reviews where variations had previously differed only by HTML formatting or trailing punctuation.
- **Actual result**:
  - Total duplicates removed: `424` rows
  - Cleaned working dataset: `49,576` rows
  - Post-cleaning class balance: `24,881` positive (50.19%) vs. `24,695` negative (49.81%)

---

## 5. Tokenization and Vocabulary

- **What was done**: Split cleaned text into word tokens via whitespace separation and constructed a frequency-ranked vocabulary capped at 30,000 tokens using **training data only**. Introduced two reserved special tokens: `<PAD>` at index 0 and `<UNK>` at index 1.
- **Why it was necessary**: Neural networks operate on numerical tensors, not strings. Deriving the vocabulary strictly from the training split is essential to prevent data leakage from validation and test sets.
- **What was observed**: The top 29,998 training words capture the overwhelming majority of word occurrences; rare terms and misspellings in validation/test sets smoothly fallback to `<UNK>`.
- **Actual result**:
  - Vocabulary size: exactly `30,000` tokens
  - Special tokens: `<PAD> = 0`, `<UNK> = 1`
  - Most frequent words: `the` (index 2), `and` (index 3), `a` (index 4), `of` (index 5)

---

## 6. Encoding and Padding

- **What was done**: Transformed lists of word tokens into lists of integer indices. Enforced a fixed sequence length of 200 tokens (`MAX_LENGTH = 200`): sequences longer than 200 tokens were truncated to their first 200 words, and shorter sequences were post-padded with the `<PAD>` token index (`0`).
- **Why it was necessary**: Batch parallelization in PyTorch requires uniform tensor shapes (`[batch_size, sequence_length]`).
- **What was observed**: A sequence length of 200 words balances computational efficiency and context retention, capturing the core sentiment of movie reviews.
- **Actual result**: Every review transformed into an exact 200-dimensional 1D tensor (`torch.int64`), grouped into batches of shape `[64, 200]`.

---

## 7. Embedding Explanation

- **What was done**: Implemented a trainable PyTorch `nn.Embedding(num_embeddings=30000, embedding_dim=128, padding_idx=0)` layer.
- **Why it was necessary**: One-hot encoding for 30,000 tokens produces an unwieldy, sparse 30,000-dimensional vector for each word. A dense embedding layer maps each token into a continuous 128-dimensional latent space where semantic relationships can be learned via backpropagation.
- **What was observed**: Setting `padding_idx=0` ensures that `<PAD>` tokens are represented by an immutable zero-vector that receives no gradient updates.
- **Actual result**:
  - Embedding parameters: $30{,}000 \times 128 = 3{,}840{,}000$ trainable weights
  - Output tensor shape for a batch: `[64, 200, 128]`

---

## 8. LSTM Architecture

- **What was done**: Built `SentimentLSTM` consisting of the 128-dim embedding layer, a single-layer LSTM with 128 hidden units (`batch_first=True`), and a linear classification head (`nn.Linear(128, 1)`).
- **Why it was necessary**: LSTMs mitigate the vanishing gradient problem in standard RNNs by maintaining an independent cell state ($c_t$) regulated by three gates: Forget Gate ($f_t$), Input Gate ($i_t$), and Output Gate ($o_t$).
- **What was observed**: The final hidden state $h_T$ of the sequence encapsulates the accumulated semantic information across all 200 timesteps.
- **Actual result**:
  - LSTM layer parameters: $4 \times (128^2 + 128^2 + 256) = 132{,}096$
  - Total trainable parameters: `3,972,225`

---

## 9. GRU Architecture

- **What was done**: Built `SentimentGRU` with identical structural dimensions: 128-dim embedding, 128 hidden units (`batch_first=True`), and `nn.Linear(128, 1)`.
- **Why it was necessary**: GRU simplifies the recurrent structure by merging the cell state and hidden state, using only two gates: a Reset Gate ($r_t$) and an Update Gate ($z_t$).
- **What was observed**: With one fewer gate per recurrent step, GRU eliminates 25% of the recurrent matrix operations, decreasing memory and compute overhead.
- **Actual result**:
  - GRU layer parameters: $3 \times (128^2 + 128^2 + 256) = 99{,}072$
  - Total trainable parameters: `3,939,201` (exactly `33,024` fewer parameters than LSTM)

---

## 10. Training Methodology

- **What was done**: Configured an end-to-end training engine (`src.training.trainer.Trainer`) utilizing Adam optimizer (`lr=0.001`), `nn.BCEWithLogitsLoss()`, batch size 64, and stratified data splits (80% train: 39,660 reviews; 10% val: 4,958 reviews; 10% test: 4,958 reviews).
- **Why it was necessary**: A controlled experimental framework guarantees that any performance variance is attributable solely to the recurrent cell architecture rather than external confounders.
- **What was observed**: Training accuracy scaled from ~51% in Epoch 1 to over 90% by Epoch 4 for both models.
- **Actual result**: Monitored training loss, validation loss, training accuracy, and validation accuracy for every epoch. Saved the checkpoint with lowest validation loss.

---

## 11. Validation and Overfitting

- **What was done**: Tracked validation loss after every epoch and implemented best-model checkpointing to prevent overfitted models from corrupting final evaluation.
- **Why it was necessary**: Recurrent models with millions of parameters can easily memorize the training corpus without learning generalizable patterns.
- **What was observed**: In an exploratory 10-epoch experiment, validation loss hit its minimum around Epoch 4 (LSTM: `0.3187`) and Epoch 5 (GRU: `0.3202`), after which validation loss increased while training loss continued downward.
- **Actual result**: Constrained final training to 5 epochs. Best LSTM model selected at Epoch 4; best GRU model selected at Epoch 5.

---

## 12. Model Evaluation

- **What was done**: Evaluated the best-performing checkpoints on the completely held-out test split of 4,958 reviews using a unified evaluation script (`evaluate_model`).
- **Why it was necessary**: Testing on previously unseen data provides an unbiased assessment of generalization capability in real-world deployment.
- **What was observed**: Both models maintained high test generalization within 1-2% of their validation accuracy scores.
- **Actual result**:
  - LSTM Test Accuracy: `86.37%`
  - GRU Test Accuracy: `85.48%`
  - Evaluation latency: `0.57 seconds` on GPU for all 4,958 reviews.

---

## 13. Metrics Explanation

- **What was done**: Measured multiple complementary evaluation metrics:
  - **Accuracy**: Overall fraction of correct predictions ($\frac{\text{TP} + \text{TN}}{\text{Total}}$).
  - **Precision**: Proportion of positive predictions that were truly positive ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
  - **Recall (Sensitivity)**: Proportion of actual positive reviews that were correctly identified ($\frac{\text{TP}}{\text{TP} + \text{FN}}$).
  - **F1-Score**: Harmonic mean of precision and recall ($\frac{2 \cdot P \cdot R}{P + R}$).
- **Why it was necessary**: Accuracy alone masks class-specific asymmetries and biases in classification behavior.
- **What was observed**: While overall accuracy was close (86.4% vs 85.5%), precision and recall broke down distinctly across the two classes.
- **Actual result**:
  - LSTM Positive F1: `86.06%` | Macro F1: `86.36%`
  - GRU Positive F1: `84.55%` | Macro F1: `85.43%`

---

## 14. Confusion Matrix Explanation

- **What was done**: Computed and plotted confusion matrices for both models on the 4,958 test reviews (2,470 negative reviews, 2,488 positive reviews).
- **Why it was necessary**: Confusion matrices break down raw prediction errors into False Positives (Type I error) and False Negatives (Type II error).
- **What was observed**:
  - **LSTM Confusion Matrix**:
    - True Negative: `2,196` | False Positive: `274`
    - False Negative: `402` | True Positive: `2,086`
    - Total errors: `676`
  - **GRU Confusion Matrix**:
    - True Negative: `2,268` | False Positive: `202`
    - False Negative: `518` | True Positive: `1,970`
    - Total errors: `720`
- **Actual result**: LSTM produced 44 fewer total misclassifications than GRU and achieved a more balanced distribution of errors.

---

## 15. LSTM vs GRU Comparison

- **What was done**: Synthesized a head-to-head empirical comparison across accuracy, parameter complexity, training speed, and error rates.
- **Why it was necessary**: To provide students and practitioners with an evidence-based answer to the classic machine learning question: *"Should I choose LSTM or GRU?"*
- **What was observed**:
  - GRU trains **16.7% faster** (34.31s vs 41.17s) and has **33,024 fewer parameters**.
  - LSTM achieves **higher overall accuracy** (+0.89%) and **substantially better positive recall** (83.84% vs 79.18%).
- **Actual result**: In terms of final predictive quality, LSTM is superior on this dataset; in terms of computational efficiency and training throughput, GRU is superior.

---

## 16. Final Findings

- **What was done**: Analyzed the full experimental record and audited the Colab notebook implementation.
- **Why it was necessary**: To ensure conclusions reflect genuine model behavior rather than artifactual quirks.
- **What was observed**:
  1. Recurrent models with trainable embeddings achieve ~86% sentiment accuracy from scratch on IMDb.
  2. The historical evaluation disparity between positive and negative recall in Colab was caused by an evaluation threshold quirk (`logit >= 0.5` instead of `logit >= 0.0`), which biased predictions toward the negative class.
  3. Because embedding parameters dominate the total model size (96.7%), the GRU's recurrent parameter savings amount to less than a 1% reduction in total model size.
- **Actual result**: LSTM selected as the primary recommended model for production classification, with GRU recognized as a competitive lightweight alternative.

---

## 17. Limitations & Future Improvements

- **What was done**: Identified technical boundaries and formulated a roadmap for further advancement.
- **Why it was necessary**: Transparent disclosure of model boundaries is a core tenet of responsible engineering.
- **What was observed**:
  - Embeddings were trained from scratch rather than initialized with GloVe or FastText.
  - No dropout or weight decay was applied, causing validation loss to plateau after 4-5 epochs.
  - Padding was static rather than dynamic with `pack_padded_sequence`.
- **Actual result / Roadmap**:
  1. Add `nn.Dropout(0.3)` to curb overfitting.
  2. Test bidirectional recurrence (`bidirectional=True`).
  3. Export model weights to ONNX format for client-side JavaScript execution in the browser via WebAssembly (`onnxruntime-web`).
  4. Launch the frontend educational demonstration website on Netlify.
