# Dataset and Preprocessing Pipeline

This document details the exact characteristics of the dataset, text normalization rules, vocabulary construction, sequence encoding, and splitting methodology.

---

## 1. Dataset Shape and Schema

- **Source File**: `data/IMDB Dataset.csv`
- **Total Rows**: Exactly `50,000`
- **Total Columns**: `2`
  - `review` (string): Full raw movie review text.
  - `sentiment` (string): Binary classification label (`positive` or `negative`).
- **File Size**: ~66.2 MB (66,212,309 bytes)

### Schema Verification
```python
df.info()
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 50000 entries, 0 to 49999
# Data columns (total 2 columns):
#  #   Column     Non-Null Count  Dtype 
# ---  ------     --------------  ----- 
#  0   review     50000 non-null  object
#  1   sentiment  50000 non-null  object
```

---

## 2. Class Distribution and Missing Values

- **Missing Values**:
  - `review`: 0 missing values (100% complete)
  - `sentiment`: 0 missing values (100% complete)
- **Class Balance**: Perfectly balanced 50/50 split in raw data:
  - `positive`: 25,000 reviews (50.00%)
  - `negative`: 25,000 reviews (50.00%)

---

## 3. Duplicate Row Detection & Cleaning

In the raw CSV:
- **Raw Exact Duplicates**: `418` duplicate rows exist where both `review` and `sentiment` are identical.
- **After Text Normalization**: An additional `6` duplicate reviews emerge because subtle differences in punctuation, case, or extra whitespace are eliminated.
- **Total Removed Duplicates**: `424` rows.
- **Cleaned Working Dataset**: `49,576` rows.

### Deduplication Impact on Class Distribution
After deduplicating the cleaned dataset:
- `positive` (label 1): `24,881` reviews (50.19%)
- `negative` (label 0): `24,695` reviews (49.81%)
- Total: `49,576` reviews.

---

## 4. Text-Cleaning Operations

Text cleaning is performed by `src.features.tokenizer.clean_text` using sequential regex transformations:

1. **Lowercasing**: Normalizes all characters to lowercase (`text.lower()`).
2. **HTML Entity Unescaping**: Converts entities like `&amp;`, `&quot;`, `&#39;` to plain text characters (`html.unescape(text)`).
3. **HTML Line Break Removal**: Replaces `<br>`, `<br/>`, and `<br />` tags with a single whitespace (`re.sub(r"<br\s*/?>", " ", text)`).
4. **HTML Tag Stripping**: Removes any remaining open/closed HTML tags (`re.sub(r"<[^>]+>", " ", text)`).
5. **URL Removal**: Strips hyperlinks starting with `http://`, `https://`, or `www.` (`re.sub(r"https?://\S+|www\.\S+", " ", text)`).
6. **Digit Removal**: Strips numeric digits (`0-9`), focusing representation on sentiment vocabulary (`re.sub(r"\d+", " ", text)`).
7. **Punctuation & Special Character Removal**: Strips all non-alphabetic characters, retaining only `a-z` and whitespace (`re.sub(r"[^a-z\s]", " ", text)`).
8. **Whitespace Normalization**: Collapses repeated whitespace, tabs, and newlines into a single space, followed by trimming leading and trailing spaces (`re.sub(r"\s+", " ", text).strip()`).

### Before / After Example
- **Raw**: `"A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. <br /><br />The actors are extremely well chosen- Michael Sheen not only "has got all the polari"..."`
- **Cleaned**: `"a wonderful little production the filming technique is very unassuming very old time bbc fashion and gives a comforting and sometimes discomforting sense of realism to the entire piece the actors are extremely well chosen michael sheen not only has got all the polari..."`

---

## 5. Tokenization and Vocabulary Construction

### Tokenization Strategy
Simple whitespace-delimited word tokenization is applied (`text.split()`). Because punctuation and digits were stripped in the text cleaning stage, every token is purely an alphabetical word.

### Special Tokens
Two designated special tokens are defined:
1. `<PAD>` (Index `0`): Used for post-padding shorter sequences. Instructs PyTorch's `nn.Embedding(padding_idx=0)` to freeze the embedding vector for index 0 to constant zeros and avoid updating its gradient during backpropagation.
2. `<UNK>` (Index `1`): Out-of-vocabulary (OOV) representation for any token absent from the learned vocabulary.

### Vocabulary Size
- `MAX_VOCAB_SIZE`: `30,000` tokens.
- Composed of:
  - `<PAD>`: index 0
  - `<UNK>`: index 1
  - Top `29,998` most frequent words from the training split.

### Top Tokens in Training Vocabulary
```text
Index 0:  <PAD>
Index 1:  <UNK>
Index 2:  the
Index 3:  and
Index 4:  a
Index 5:  of
Index 6:  to
Index 7:  is
Index 8:  it
Index 9:  in
Index 10: i
Index 11: this
Index 12: that
Index 13: s
Index 14: was
```

---

## 6. Sequence Encoding, Truncation, and Padding

Recurrent neural networks require uniform tensor dimensions per batch.
- **Target Sequence Length (`MAX_LENGTH`)**: `200` tokens.
- **Truncation Policy**: Sequences exceeding 200 tokens are truncated to the first 200 tokens (`ids[:max_length]`).
- **Padding Policy**: Sequences under 200 tokens are post-padded with the `<PAD>` token index (`0`) up to length 200 (`ids += [0] * (max_length - len(ids))`).

### Output Tensor Format
- Each review is encoded as a 1D integer tensor: `torch.tensor(encoded, dtype=torch.long)` of length 200.
- Labels are encoded as 0D/1D float tensors: `torch.tensor(label, dtype=torch.float32)` where `0.0` = negative, `1.0` = positive.
- A batch of size 64 produces:
  - Input reviews: `torch.Size([64, 200])` (dtype: `torch.int64`)
  - Target labels: `torch.Size([64])` (dtype: `torch.float32`)

---

## 7. Train / Validation / Test Splitting

To ensure unbiased evaluation, the cleaned dataset (49,576 rows) is partitioned into three independent subsets:

| Split | Percentage | Review Count | Negative Labels (0) | Positive Labels (1) |
|---|---|---|---|---|
| **Training** | 80% | **39,660** | 19,756 (49.81%) | 19,904 (50.19%) |
| **Validation** | 10% | **4,958** | 2,470 (49.82%) | 2,488 (50.18%) |
| **Test** | 10% | **4,958** | 2,469 (49.80%) | 2,489 (50.20%) |
| **Total** | 100% | **49,576** | 24,695 | 24,881 |

### Splitting Method
- Stratified sampling using `sklearn.model_selection.train_test_split` with `stratify=df['label']`.
- Deterministic random seed: `random_state = 42`.
- Two-stage split:
  1. $80\%$ Train ($39,660$), $20\%$ Temporary ($9,916$).
  2. Temporary split $50/50$ into Validation ($4,958$) and Test ($4,958$).

---

## 8. Data Leakage Prevention

To ensure strict experimental integrity:
1. **Vocabulary Isolation**: The vocabulary and word frequency counter are derived **exclusively from the 39,660 training reviews**. Validation and test reviews are never scanned during vocabulary creation. Any rare word in the validation or test set that did not appear in the training top 29,998 words is mapped to `<UNK>` (index 1).
2. **Zero Information Leakage in Normalization**: Preprocessing parameters (regex cleaning rules) are stateless and purely deterministic.
3. **Independent Evaluation**: The test split is held out completely during training, validation loss monitoring, early stopping, and checkpoint selection.
