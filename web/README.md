# IMDb sentiment-analysis website

Static portfolio site for the repository's LSTM-versus-GRU experiment. It uses plain HTML, CSS, and JavaScript; no build process or backend is needed.

## Run locally

From the repository root:

```powershell
python -m http.server 8000 --directory web
```

Open `http://localhost:8000`. Serving it avoids browser restrictions that can affect local files.

## Structure

- `index.html` — semantic long-form project presentation.
- `css/styles.css` — design tokens, responsive layout, and accessibility styles.
- `js/charts.js` — the single configuration point for verified historical benchmark metrics and rendered SVG charts.
- `js/demo.js` — input validation and the honest inference-unavailable UI.
- `js/main.js` — mobile navigation and active-section state.

## Data integrity

The training charts use recorded five-epoch historical Colab values documented in `docs/EXPERIMENTS_AND_FINDINGS.md`. The test comparison uses the historical 4,958-review benchmark. The page labels both explicitly and discloses the notebook's historical raw-logit threshold inconsistency.

`results/training_history.csv` and `results/evaluation_results.json` are a separate local 50-sample diagnostic run, so they are not substituted for the full benchmark.

## Inference status

The repository has PyTorch `.pt` checkpoints and `artifacts/vocab.json`, but no ONNX/TensorFlow.js artifact or browser runtime. The demo deliberately cannot produce a prediction. To enable real browser inference, export and validate each checkpoint to ONNX; ship an appropriate browser runtime; reproduce `clean_text`, vocabulary lookup, post-padding and truncation at length 200; and validate browser outputs against `predict.py` on identical inputs.

## Known limitations

The font import requires network access; system fallbacks remain available. No public Git remote is configured in this working copy, so the site intentionally displays no invented repository URL.
