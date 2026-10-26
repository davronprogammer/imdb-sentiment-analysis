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

The repository has two PyTorch checkpoints (`lstm_best.pt`, `gru_best.pt`) and `artifacts/vocab.json`, but currently has no checked-in ONNX artifacts. The browser-readable vocabulary and preprocessing contract are present at `assets/models/vocabulary.json` and `assets/models/preprocessing-config.json`.

Until validated ONNX files exist, the interactive demo uses a small transparent lexicon-based baseline that runs entirely in the visitor's browser. It counts fixed positive and negative review terms, reports the signed lexical score and matched words, and labels the result as a baseline. It is not an LSTM/GRU prediction, does not claim model accuracy, and does not display a confidence percentage. No review is sent to a server.

The exact Python preprocessing contract is: lowercase; HTML entity/tag removal; URL, digit, and non-`a-z` removal; whitespace collapse; whitespace tokenization; training-vocabulary lookup (`<PAD>=0`, `<UNK>=1`); truncate to 200; and post-pad with zero. Models consume `int64` tensor input `[batch, 200]` and return one raw logit per review. Apply sigmoid for confidence and use `logit >= 0` for a positive label.

### Export and validate ONNX

The export helper uses real checkpoints only and writes models to `web/assets/models/`:

```powershell
pip install onnx onnxruntime
python scripts/export_onnx.py --validate
```

`--validate` compares ONNX Runtime logits with PyTorch logits for both models and fails on a numerical mismatch. Only after that succeeds should a browser runtime such as `onnxruntime-web` be connected to the demo. Models should be lazy-loaded only after the visitor selects one.

Training, export, browser inference, and hosting are separate: Python trains/exports weights; the browser runs an exported model locally; Netlify serves static files only and never trains or classifies reviews server-side.

## Netlify deployment

`netlify.toml` publishes `web/` with no build command. In Netlify, create a site from this repository, leave the build command blank, and set the publish directory to `web` (the committed configuration supplies the same value). Deploy. The model cache header applies only once versioned ONNX files have been generated; no SPA fallback or backend is configured.

## Image inventory

`assets/images/` contains the hero film background and four editorial content images: `imdb-dataset-editorial.png`, `data-preparation.png`, `recurrent-networks-study.png`, and `sentiment-experiment-lab.png`. All content images are loaded lazily with descriptive alternative text.

## Troubleshooting

- If `python` resolves to a Windows Store alias, install or activate a real Python environment first.
- If export says `onnxruntime` is missing, install the optional packages shown above.
- If validation fails, do not deploy the generated model; investigate architecture/checkpoint compatibility first.
- If a model is large for a mobile connection, export only the required architecture or use a separately validated quantization workflow.

## Known limitations

The lexicon baseline applies limited negation handling, but cannot reliably interpret context, irony, or vocabulary outside its short term list; it is a usable fallback, not a replacement for the trained models. The font import requires network access; system fallbacks remain available. The configured public remote is linked from the website. This environment’s `python.exe` and `py.exe` are Windows Store aliases and cannot execute, so ONNX export and Python tests require a working local interpreter.
