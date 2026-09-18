# AgenticXAI

An agentic framework for black-box explainability across classical ML and generative models. The end goal: an agent that reads a model's API description, infers what kind of model it is, generates synthetic data where needed, runs the right explainability techniques (LIME, PDP, Permutation Importance for tabular models; leave-one-out occlusion and self-explanation prompting for LLMs), validates the results, and produces a report - without needing white-box access to the model.

This project is motivated by the RBI's *Guidance on Regulatory Principles for Model Risk Management, 2026*, which requires explainability and risk assessment for models used by regulated financial entities, including third-party/vendor models where internal access is typically unavailable.

## Project structure

```
.
├── demo_models/                        # Dataset + model building, for demo purposes only
│   ├── datasets/                        # Downloaded datasets (house_prices, bank_marketing, bbc_news)
│   ├── checkpoints/                     # Trained model artifacts, metrics, feature stats
│   ├── download_datasets.py             # Pulls all 3 datasets via curl
│   ├── house_prices_train.py            # Preprocessing + ElasticNet training (regression)
│   ├── house_prices_rf_train.py         # Random Forest (evaluated, not used - see Results)
│   ├── house_prices_gb_train.py         # Gradient Boosting (evaluated, not used - see Results)
│   ├── house_prices_poly_train.py       # Polynomial features + ElasticNet (evaluated, not used)
│   ├── house_prices_ann_train.py        # 2-layer ANN (kept alongside ElasticNet)
│   ├── bank_marketing_train.py          # Preprocessing + Logistic Regression (classification)
│   ├── bank_marketing_rf_train.py       # Random Forest classifier (reuses same splits)
│   ├── bank_marketing_ann_train.py      # 2-layer ANN classifier (reuses same splits)
│   └── bbc_news_train.py                # Text classification, fine-tuned encoder (in progress)
│
├── demo_serving/                        # Model serving API
│   ├── app.py                            # FastAPI app: predict + describe endpoints, both datasets
│   ├── run_server.sh                     # Starts the server
│   └── test.py                           # Hits the running server with sample data points
│
├── src/
│   ├── xai/                              # Core, from-scratch explainability tools
│   │   ├── errors.py                      # Custom exceptions for feature_stats schema violations
│   │   ├── schema_utils.py                # Shared validation + numeric range/fallback resolution
│   │   ├── lime.py                        # From-scratch LIME (raw feature space, weighted local surrogate)
│   │   ├── pdp.py                         # From-scratch PDP (single feature at a time)
│   │   └── tests/
│   │       ├── test_lime.py               # LIME on hardcoded reg/classification datasets
│   │       ├── test_pdp.py                # PDP on the same datasets, all features
│   │       └── house_pred_lime_pdp.py     # LIME + PDP against the live serving API
│   │
│   └── pipeline/                         # Deterministic agentic reporting pipeline
│       ├── llm_client.py                  # OpenAI wrapper (text + JSON-mode completions)
│       ├── pdf_export.py                  # Markdown -> PDF (pandoc + wkhtmltopdf)
│       ├── output_handler.py              # Abstract persistence boundary (no concrete impl here)
│       ├── run_pipeline.py                # Functional entry point - probes /describe, dispatches
│       │                                    to a modality backend by BACKENDS registry (no CLI)
│       ├── tabular/                       # Tabular (numeric+categorical) modality backend
│       │   ├── model_client.py             # Parses /describe, generic predict_fn, synthetic data
│       │   ├── explain.py                   # LIME importance ranking, PDP, sample-case walkthroughs
│       │   ├── report.py                     # Chart generation, LLM report writing
│       │   └── backend.py                    # Orchestrates the above into one run(...) call
│       └── text/                          # Placeholder for a future text-modality backend
│
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Task | Features | Status |
|---|---|---|---|
| House Prices (`yasserh/housing-prices-dataset`) | Regression | Numeric + categorical | Done |
| Bank Marketing (`janiobachmann/bank-marketing-dataset`) | Classification | Numeric + categorical | Done |
| BBC News (GitHub CSV) | Text classification | Raw text | In progress |

## Preprocessing pipelines

### House Prices

1. Drop missing values.
2. One-hot encode categorical features (all categories kept, fit on train split only to avoid leakage).
3. Split 70/10/20 (train/val/test), val reserved for hyperparameter tuning.
4. Standardize numeric features and the target using train-split mean/std.
5. Save `train.csv`, `val.csv`, `test.csv` and `feature_stats.json` (numeric mean/std/min/max/p25/p75, categorical categories + frequencies, target mean/std).

### Bank Marketing

1. Drop missing values.
2. Drop `duration` - a well-known leakage feature in this dataset (call duration correlates almost mechanically with the target).
3. Derive `previously_contacted` (1 if `pdays != -1`, else 0) before treating `pdays` as an ordinary numeric feature, so the `-1` "never contacted" sentinel isn't perturbed downstream as if it were a real continuous value.
4. Encode target (`deposit`: yes/no -> 1/0).
5. Stratified split 70/10/20 (preserves class balance across splits).
6. One-hot encode categorical features (fit on train split only), standardize numeric features.
7. Save `train.csv`, `val.csv`, `test.csv` and `feature_stats.json` (target stored as `positive_class` + `encoding`, not mean/std, since it's classification).

Every downstream model script (RF, ANN, for both datasets) reuses these exact splits and feature stats with no re-encoding, and guards against being run before the base training script with a clear error message.

## Results

### House Prices (regression, test R²)

| Model | Test R² | Notes |
|---|---|---|
| ElasticNet (L1+L2) | **0.679** | Final choice |
| 2-layer ANN | 0.65 | Kept alongside ElasticNet |
| Random Forest | lower | Evaluated, dropped |
| Gradient Boosting | lower | Evaluated, dropped |
| Polynomial + ElasticNet | lower | Evaluated, dropped |

At this dataset size (~545 rows), more flexible models consistently failed to beat a well-regularized linear model - consistent with published benchmarks showing the gap between tree ensembles/deep models and linear baselines shrinks (and can invert) on small tabular datasets.

### Bank Marketing (classification, test accuracy / F1)

| Model | Test Accuracy | Test F1 |
|---|---|---|
| Logistic Regression | *(run on real data to fill in)* | |
| Random Forest | | |
| 2-layer ANN | | |

Tuned via validation-set F1 (not accuracy), since bank marketing datasets are typically imbalanced toward "no".

## Serving API

A single FastAPI app serves both House Price and Bank Marketing models. Each model exposes two endpoints, under its own dataset prefix:

- `POST /{dataset}/{model_key}/predict` - takes raw feature values (not preprocessed), returns a prediction in original units (`predicted_price` for regression; `predicted_class` + `predicted_probabilities` (one entry per class) + `confidence` for classification - binary and multi-class endpoints return the same shape).
- `GET /{dataset}/{model_key}/describe` - returns job type, per-feature schema (numeric stats or categorical categories/frequencies, target excluded), an example request body, and which output field to read. For classification, also declares `reference_class` - the one class whose probability stays fixed as the explainability target (the dataset's positive class for binary; the training-majority class for multi-class, since there's no natural "positive" class).

| Dataset | model_key options |
|---|---|
| `house_prices` | `linear`, `ann` |
| `bank_marketing` | `logistic`, `rf`, `ann` |

Both dataset families share the same generic model-loading and feature-vector-building logic underneath (driven entirely by `feature_stats.json`'s shape) - adding a new dataset means one new Pydantic schema + two route functions, no changes to shared logic.

### Running it

```bash
cd demo_serving
./run_server.sh
```

In a separate terminal:

```bash
python test.py
```

## Explainability core (`src/xai/`)

From-scratch implementations, operating entirely in raw feature space (perturb real-world values, call a `predict_fn`, never touch model internals) - this is what makes the whole approach work against a served API, not just a local model object.

- **LIME** (`lime.py`): perturbs one instance (numeric: Gaussian noise scaled by training std; categorical: swap to a different category sampled by training frequency), fits a weighted local linear surrogate, returns per-feature attributions plus the surrogate's own R² as a built-in confidence signal (low R² = don't trust this explanation).
- **PDP** (`pdp.py`): one feature at a time - fixes a feature to a grid of candidate values across a real background batch, averages the model's output at each value. Numeric grids use `p25/p75` -> `min/max` -> `mean +/- 2*std` as a fallback hierarchy, depending on what's available in `feature_stats`.
- **Schema validation** (`schema_utils.py`, `errors.py`): hard failure on missing categorical categories or missing numeric mean/std (no guessing); graceful fallback only for optional distribution shape stats.

All three are tested against small, hand-crafted datasets with a known ground-truth relationship (not just random data), so outputs can be sanity-checked against intuition, not just checked for "did it crash." Also validated end-to-end against the live House Price serving API (`house_pred_lime_pdp.py`).

## Agentic reporting pipeline (`src/pipeline/`)

A fully deterministic pipeline (no agent decision-making - every step is fixed) that goes from "here's a model's URL" to "here's a PDF explainability report," using an LLM only for two swappable steps: synthetic data generation and report writing.

1. **Probe** the model's `/describe` endpoint - learns job type, feature schema, and which field to read from `/predict` responses.
2. **Generate synthetic data** via LLM (JSON-mode, schema-aware prompt), validate every row's schema/bounds, retry only the shortfall (not the whole batch) up to 3 times - if still short, proceed with whatever valid rows were collected rather than discarding them.
3. **Build a generic `predict_fn`** that works for regression and classification alike, self-correcting on the API's own validation errors (rounds int-typed fields, clips range violations) rather than guessing in advance.
4. **Run LIME across every synthetic point**, average `|attribution|` per feature, rank top 5 numeric + top 5 categorical, normalize to a 0-1 proportional share of the total.
5. **Run PDP** on the top 4 numeric + top 4 categorical features, reusing the synthetic batch as background data.
6. **Build 2 individual sample-case walkthroughs** (real input values + real predicted output, inserted deterministically - never LLM-transcribed) by reusing LIME results already computed in step 4.
7. **Generate the report via LLM**, with three safety mechanisms:
   - the LLM only ever places text markers, never file paths - chart images are substituted in afterward, so a broken/hallucinated path is impossible;
   - the words "LIME," "PDP," etc. never appear in the JSON the LLM is given, and the system prompt explicitly forbids naming any specific technique - findings are described as "sensitivity" and "impact," not disclosed methodology;
   - the factual header (model tested, endpoint, task type, timestamp) is generated by code, never by the LLM.
8. **Convert to PDF** (`pandoc` + `wkhtmltopdf`). Side-by-side chart layouts use HTML `<table>` cells, not CSS flexbox - `wkhtmltopdf`'s old rendering engine doesn't support flexbox, so a layout that looks correct in a markdown preview can silently collapse in the PDF if flexbox is used.

`run_pipeline.py` has no CLI - it's a plain function meant to be imported and called (by a microservice, a script, a REPL), taking an `OutputHandler` that decides where generated artifacts (report PDF/JSON/markdown, charts) end up; no concrete `OutputHandler` ships in this repo. See `src/pipeline/output_handler.py` and `CLAUDE.md` for the contract and an example call.

### Running it

```bash
export OPENAI_API_KEY=sk-...
cd demo_serving && ./run.sh                    # separate terminal
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline.run_pipeline import run_pipeline
from pipeline.output_handler import OutputHandler
# ...implement an OutputHandler.persist(working_dir, artifact_filenames) -> {key: final_path}...
print(run_pipeline('http://localhost:8000/house_prices', 'linear', my_output_handler))
"
```

## Status

- [x] House Prices: preprocessing, model comparison, final models (ElasticNet + ANN), serving API
- [x] Bank Marketing: preprocessing, Logistic Regression + RF + ANN, serving API
- [ ] BBC News: fine-tuned encoder text classification model
- [x] `src/xai/`: LIME, PDP, schema validation - tested on hardcoded datasets and the live serving API
- [x] `src/pipeline/`: full deterministic pipeline (probe -> synthetic data -> LIME/PDP -> report -> PDF)
- [ ] Text explainability (leave-one-out occlusion, self-explanation prompting) for BBC News
- [ ] Permutation Importance tool (label-free / prediction-shift variant, discussed but not yet built)