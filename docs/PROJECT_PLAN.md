# Project Plan — IMDB Sentiment Classification (LSTM vs. GRU)

## 1. Project Objective

The goal of this project is to build, evaluate, and comparatively analyze two recurrent neural network architectures—**Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)**—for binary sentiment classification on the standard IMDb Large Movie Review dataset (50,000 reviews). The project emphasizes:
- Mathematical and architectural understanding of recurrent units.
- Strict data-leakage prevention across data processing and vocabulary formation.
- Reproducible PyTorch engineering pipelines.
- Transparent reporting distinguishing historical GPU benchmarks from CPU local environments.
- Preparing rich educational content and artifacts for a future frontend showcase.

---

## 2. Scope & Boundaries

### In-Scope
- Modularizing the educational Google Colab notebook into clean, maintainable Python modules (`src/`).
- Dataset auditing (verifying row counts, class balance, missing values, duplicate detection).
- Building an end-to-end reproducible pipeline: text normalization, vocabulary construction, integer sequence encoding, post-padding, PyTorch datasets, and DataLoaders.
- Re-implementing PyTorch LSTM and GRU models with identical structural hyperparameters.
- Configurable training engine with validation tracking, checkpointing, and early stopping.
- Comprehensive metric evaluation: Accuracy, Precision, Recall, F1-score (positive, negative, macro, weighted), confusion matrices, and publication-quality plots.
- Complete documentation suite detailing dataset characteristics, model mathematics, Colab audit findings, and educational website copy.
- Comprehensive automated test suite with pytest.

### Out-of-Scope (Strict Non-Goals)
- Building the frontend website (HTML, CSS, JS) during this phase.
- Backend API servers (FastAPI/Flask) or database layers.
- Pretrained language models (BERT, RoBERTa) or GloVe embeddings, preserving the original educational focus on foundational RNN mechanics.

---

## 3. Completed Work

1. **Repository & Notebook Audit**:
   - Analyzed `notebooks/sentiment_classification.ipynb` (57 cells).
   - Audited `data/IMDB Dataset.csv` (50,000 rows, 25k positive / 25k negative, 418 raw duplicates, 424 post-cleaning duplicates, leaving 49,576 unique reviews).
   - Uncovered critical evaluation bug in original notebook: cell 54 compared raw logits directly to `>= 0.5` instead of `>= 0.0` (or `sigmoid >= 0.5`), skewing recall metrics toward negative predictions.
2. **Modular Architecture Implementation (`src/`)**:
   - `src/features/tokenizer.py`: Text cleaning regex, tokenization.
   - `src/features/vocabulary.py`: Special tokens (`<PAD>`, `<UNK>`), training-only build, serialization.
   - `src/features/encoding.py`: Token indexing, truncation, post-padding.
   - `src/data/loader.py`: CSV loading, deduplication, label mapping, stratified splitting.
   - `src/data/dataset.py`: PyTorch `IMDBDataset` and DataLoader factory.
   - `src/models/lstm.py` & `src/models/gru.py`: PyTorch recurrent classifiers.
   - `src/models/base.py`: Parameter counting and model metadata derivation.
   - `src/training/trainer.py`: Reusable training loop, early stopping, best checkpointing.
   - `src/evaluation/`: Metrics calculation, prediction generation, evaluation runner, and plotting.
   - `src/utils/`: Configuration dataclass, random seed helper, and I/O utilities.
3. **CLI Scripts**:
   - `train.py`: Unified pipeline for training, evaluation, plot generation, and metric reporting.
   - `evaluate.py`: Standalone evaluation of saved checkpoints.
   - `predict.py`: Interactive and CLI single-sentence sentiment inference.
4. **Test Suite**:
   - Implemented 9 unit tests in `tests/test_pipeline.py` verifying preprocessing, special tokens, OOV handling, padding, label mapping, split leakage, tensor shapes, parameter counts, and metrics.
   - 100% test pass rate achieved under `pytest`.
5. **Diagnostic Pipeline Execution**:
   - Validated end-to-end execution on local CPU, producing all required machine-readable tables and visualizations in `results/`.

---

## 4. Remaining Work & Future Roadmap

1. **Full-Scale GPU Retraining**:
   - Execute 5 full epochs on the complete 39,660 training split using CUDA acceleration to generate locally verified final weights matching Colab performance.
2. **ONNX / WebAssembly Export**:
   - Export best checkpoint to ONNX format and quantize (int8) for sub-10MB in-browser client-side inference using `onnxruntime-web`.
3. **Frontend Implementation (Phase 2)**:
   - Design and build the interactive educational website (HTML5, modern CSS, vanilla ES6+ JavaScript) using the factual copy in `docs/WEBSITE_CONTENT.md`.
   - Host as a zero-backend static site on Netlify or GitHub Pages.

---

## 5. Technical Decisions & Rationale

| Decision | Selection | Rationale |
|---|---|---|
| **Loss Function** | `nn.BCEWithLogitsLoss` | Numerically stable log-sum-exp formulation prevents underflow/overflow compared to separate `nn.Sigmoid` + `nn.BCELoss`. |
| **Logit Decision Boundary** | `logit >= 0.0` | In `BCEWithLogitsLoss`, $\sigma(0.0) = 0.5$. Fixes historical Colab evaluation threshold anomaly (`output >= 0.5`). |
| **Vocabulary Construction** | Train Split Only | Building vocabulary strictly from the 80% training split prevents out-of-sample data leakage. |
| **Embedding Strategy** | Trainable `nn.Embedding` (128-dim) | Keeps models self-contained without external 1GB pretrained vectors, illustrating raw representation learning. |
| **Padding Strategy** | Post-padding with `<PAD>=0` | Preserves sequence alignment while informing `nn.Embedding(padding_idx=0)` to zero out gradient updates on padded positions. |
| **Random State** | `seed = 42` | Consistent random seed across Python, NumPy, Scikit-learn, and PyTorch CPU/GPU ensures deterministic reproducibility. |

---

## 6. Final Deliverables

- `src/`: Complete modularized Python package.
- `train.py`, `evaluate.py`, `predict.py`: Usable command-line interfaces.
- `tests/test_pipeline.py`: Comprehensive test suite.
- `results/`: Comparison tables (CSV/JSON), training histories, classification reports, and PNG plots.
- `artifacts/`: Vocabulary JSON and PyTorch `.pt` checkpoints.
- `docs/`: In-depth documentation suite (`PROJECT_PLAN.md`, `DATASET_AND_PREPROCESSING.md`, `MODEL_ARCHITECTURE.md`, `EXPERIMENTS_AND_FINDINGS.md`, `MODEL_COMPARISON.md`, `WEBSITE_CONTENT.md`).
- `README.md`: Professional project landing page.
