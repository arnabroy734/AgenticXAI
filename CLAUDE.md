# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agentic framework for black-box explainability across classical ML and generative models. Given only a model's API (a `/describe` schema endpoint + a `/predict` endpoint), it infers the model type, generates synthetic data, runs explainability techniques (LIME, PDP; permutation importance planned), validates results, and produces a PDF report — with no white-box access to the model. Motivated by RBI's *Guidance on Regulatory Principles for Model Risk Management, 2026* (explainability requirements for regulated financial entities' vendor/third-party models).

Two independent halves:
- `src/xai/` — from-scratch, black-box explainability primitives (LIME, PDP). Work against *any* `predict_fn`, including one backed by a live HTTP API — they never touch model internals.
- `src/pipeline/` — a fully deterministic orchestration pipeline (no agent decision-making) that chains `src/xai/` tools together, using an LLM only for two swappable steps: synthetic data generation and final report writing.

`demo_models/` and `demo_serving/` exist only to have something to point the pipeline at — they are not part of the explainability framework itself.

## Commands

```bash
# Start the demo serving API (from demo_serving/)
cd demo_serving && ./run.sh                   # uvicorn app:app on 0.0.0.0:8000

# Run the xai smoke tests (plain scripts, not pytest — run directly)
python src/xai/tests/test_lime.py
python src/xai/tests/test_pdp.py
python src/xai/tests/test_lime_text.py        # requires the serving API running (BBC News model)
python src/xai/tests/house_pred_lime_pdp.py   # LIME + PDP against the live House Price API

# Run the full reporting pipeline end to end (requires the serving API running)
# There is no CLI for this - run_pipeline() is a plain function, meant to be
# imported and called (by a microservice, a script, a REPL). It takes an
# OutputHandler instance (see src/pipeline/output_handler.py) that decides
# where generated artifacts end up; nothing ships a concrete one in this repo.
export OPENAI_API_KEY=sk-...
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline.run_pipeline import run_pipeline
from pipeline.output_handler import OutputHandler

class LocalOutputHandler(OutputHandler):
    def persist(self, working_dir, artifact_filenames):
        import shutil, os
        os.makedirs('explainability_reports/example', exist_ok=True)
        return {k: shutil.copy(os.path.join(working_dir, v), 'explainability_reports/example') for k, v in artifact_filenames.items()}

print(run_pipeline('http://0.0.0.0:8000/house_prices', 'linear', LocalOutputHandler()))
"

# Retrain a demo model (from demo_models/) — download data first, once
python demo_models/download_datasets.py
python demo_models/house_prices_train.py
python demo_models/bank_marketing_train.py
```

There is no lint/format/typecheck tooling configured, and tests are standalone scripts (no pytest runner) — run the file directly with `python`, they write JSON/PNG output into a sibling `outputs/` directory and exit nonzero-free on success (no assert-based pass/fail signal beyond that).

PDF export (`src/pipeline/pdf_export.py`) shells out to `pandoc` and `wkhtmltopdf` — both must be installed on PATH for the pipeline's final step to succeed.

`src/pipeline/` is organized by modality: `tabular/` holds everything that orchestrates numeric+categorical models end to end (this is what `run_pipeline.py` actually dispatches to today); `text/` is currently just a placeholder — BBC News explainability exists only as the standalone `src/xai/tests/test_lime_text.py` smoke test, with no synthetic-data/report/PDF orchestration yet.

## Architecture

### `src/xai/` — black-box explainability primitives

Everything operates in **raw feature space**: perturb real-world values, call a `predict_fn`, read the output. This is what lets the same code explain a local sklearn model or a remote FastAPI-served one — the tools never see model internals, only `feature_stats.json`-shaped schema plus a callable.

- **`lime.py`** — perturbs one instance (numeric: Gaussian noise scaled by training std; categorical: swap to a different category sampled by training frequency), fits a weighted local linear surrogate, returns per-feature attributions plus the surrogate's own R² as a built-in confidence signal (low R² = don't trust this explanation).
- **`lime_text.py`** — hierarchical LIME for raw text: sentence-level shading + n-gram-level highlighting within the top sentence, rendered as standalone HTML (a bar chart of n-gram names wouldn't show *where* in the text the signal is).
- **`pdp.py`** — one feature at a time: fixes a feature to a grid of candidate values across a real background batch, averages the model's output at each value. Numeric grids fall back through `p25/p75` → `min/max` → `mean ± 2*std`, depending on what's available in `feature_stats`.
- **`schema_utils.py` / `errors.py`** — hard failure on missing categorical categories or missing numeric mean/std (no silent guessing); graceful fallback only for optional distribution-shape stats.

All three are tested against small, hand-crafted datasets with a *known* ground-truth relationship (not random data), so outputs are sanity-checked against intuition rather than just "did it crash."

### `src/pipeline/` — deterministic agentic reporting pipeline

`run_pipeline.py` is a plain function (`run_pipeline(base_url, model_key, output_handler, num_points=100, num_lime_samples=500) -> dict`) meant to be imported and called — e.g. by a microservice — never a CLI; there's no `argv` parsing anywhere in this package. It:

1. Fetches `/describe` once and reads `modality` (defaulting to `"tabular"`) to pick a **backend** from the `BACKENDS` registry (`{"tabular": tabular.backend.run, ...}`). Adding a genuinely new model modality later means writing a new backend module with the same `run(...)` contract (see `tabular/backend.py`'s docstring) and adding one line to `BACKENDS` — not editing this dispatch logic.
2. Creates a private `tempfile.mkdtemp()` working directory, hands it to the chosen backend, and deletes it in a `finally` once persistence is done — so nothing lingers on disk across requests in a long-running service.
3. Calls the caller-supplied `output_handler: OutputHandler` (`output_handler.py` — an `ABC`, no concrete implementation ships here) to persist whatever the backend produced from that working directory to wherever it should finally live (local disk for a test callback, a POST to another service in production, etc.), then merges its return value into `run_pipeline`'s own result dict under `"artifacts"`. Exceptions from any step (probe, backend, handler) propagate uncaught — the caller decides how to map them to a response.

The tabular backend (`tabular/backend.py`) is today's only implementation, every step fixed logic except two which call an LLM (`llm_client.py`, OpenAI wrapper, requires `OPENAI_API_KEY`):

1. **Parse `/describe`** (`tabular/model_client.py: parse_describe_body`) — feature schema, which field to read from `/predict`, and (for classification) the fixed `reference_class` whose probability is explained (see below).
2. **Generate synthetic data** (`tabular/model_client.py: generate_synthetic_data`) via LLM JSON-mode, schema-aware prompt; validates every row's schema/bounds; retries only the shortfall (not the whole batch), up to 3 times; if still short, proceeds with whatever valid rows were collected rather than discarding them.
3. **Build a generic `predict_fn`** (`tabular/model_client.py: make_predict_fn`) that works for regression and classification alike, self-correcting on the API's own validation errors (rounds int-typed fields, clips range violations) rather than guessing schema in advance.
4. **Run LIME across every synthetic point** (`tabular/explain.py: compute_average_importance`), average `|attribution|` per feature, rank top 5 numeric + top 5 categorical, normalize to a 0–1 proportional share.
5. **Run PDP** (`tabular/explain.py: run_pdp_on_top_features`) on the top 4 numeric + top 4 categorical features, reusing the synthetic batch as background data.
6. **Build 2 sample-case walkthroughs** (`tabular/explain.py: build_sample_cases`) using real input values + real predicted output inserted deterministically (never LLM-transcribed), reusing the LIME results from step 4.
7. **Generate the report** (`tabular/report.py`) via LLM, with three safety mechanisms:
   - the LLM only ever places text markers, never file paths — chart images are substituted in afterward, so a broken/hallucinated path is impossible;
   - technique names ("LIME," "PDP," etc.) never appear in the JSON given to the LLM, and the system prompt forbids naming any specific technique — findings are described as "sensitivity" and "impact," not disclosed methodology;
   - the factual header (model tested, endpoint, task type, reference_class for classification, timestamp) is generated by code, never by the LLM.
8. **Convert to PDF** (`pdf_export.py`) via `pandoc` + `wkhtmltopdf`. Side-by-side chart layouts use HTML `<table>` cells, not CSS flexbox — `wkhtmltopdf`'s old rendering engine doesn't support flexbox, so a layout that looks fine in a markdown preview can silently collapse in the PDF.

Every artifact filename the backend produces is recorded under a generic key (`report_pdf`, `report_json`, `report_markdown`, `sensitivity_chart`, `impact_chart_<feature>`, `sample_case_<n>`) in `artifact_filenames` — never a backend-specific step name — so `OutputHandler.persist(...)` and any future backend agree on one shape.

**Classification target is fixed, not a moving argmax.** For classification jobs, LIME/PDP always explain the probability of one class named by the API's `/describe` (`output.reference_class`) — never "whichever class is currently predicted." A moving target would make the local LIME surrogate fit poorly and PDP curves show artificial cliffs whenever a perturbation flips the argmax, which is an artifact of the metric switching what it tracks, not real model behavior. `make_predict_fn` reads `response[output_field][reference_class]` for classification (a fixed dict key) vs. the flat `response[output_field]` for regression.

### `demo_serving/` — serving API for the demo models

One FastAPI app (`app.py`) serves all three demo model families under per-dataset prefixes:

- `POST /{dataset}/{model_key}/predict` — raw feature values in, prediction in original units out. Regression: `predicted_price`. Classification (binary and multi-class alike, same shape): `predicted_class`, `predicted_probabilities` (dict, one entry per class), `confidence` (display-only, probability of whichever class was actually predicted).
- `GET /{dataset}/{model_key}/describe` — job type, `modality` (`"tabular"` or `"text"` — this is what `run_pipeline.py` reads to pick a backend), per-feature schema (numeric stats or categorical categories/frequencies, target excluded), an example request body, and which output field to read. For classification, also `output.reference_class` — the one class whose probability `src/pipeline/` treats as the fixed explainability target (bank_marketing: the dataset's `positive_class`; bbc_news: statically declared per `model_key` in `BBC_NEWS_MODELS`, since a 5-way problem has no natural "positive" class — currently `"sport"`, the training-majority category).

| Dataset | model_key options |
|---|---|
| `house_prices` | `linear`, `ann` |
| `bank_marketing` | `logistic`, `rf`, `ann` |
| `bbc_news` | `tfidf`, `encoder` |

The tabular datasets (house_prices, bank_marketing) share generic model-loading and feature-vector-building logic, driven entirely by `feature_stats.json`'s shape — adding a new tabular dataset means one new Pydantic schema + two route functions, no changes to shared logic. BBC News is a different modality (raw text, multi-class) and gets its own loading/prediction path.

### `demo_models/` — training scripts, for demo purposes only

Preprocessing conventions that downstream models rely on:

**House Prices**: drop missing values → one-hot encode categoricals (fit on train split only) → 70/10/20 split → standardize numeric features + target using train-split mean/std → save `feature_stats.json` (numeric mean/std/min/max/p25/p75, categorical categories+frequencies, target mean/std).

**Bank Marketing**: drop missing values → drop `duration` (leakage: correlates almost mechanically with the target) → derive `previously_contacted` (1 if `pdays != -1`, else 0) *before* treating `pdays` as an ordinary numeric feature, so the `-1` "never contacted" sentinel isn't perturbed downstream as if it were real continuous data → encode target → **stratified** 70/10/20 split → one-hot encode + standardize → save `feature_stats.json` (`positive_class` + `encoding` instead of mean/std, since it's classification).

Every downstream model script (RF, ANN variants, for both datasets) reuses the exact same splits and `feature_stats.json` with no re-encoding, and hard-fails with a clear error if run before the base preprocessing script.

Model choice note: at this dataset size (~545 rows for house prices), more flexible models (RF, GB, polynomial features) consistently lost to a well-regularized linear model (ElasticNet, test R² 0.679) — kept ANN alongside it, dropped the others.

## Status (see README.md for the fuller table)

- BBC News: fine-tuned encoder text classification is in progress.
- Text explainability (leave-one-out occlusion, self-explanation prompting) for BBC News: not yet built.
- Permutation Importance tool (label-free / prediction-shift variant): discussed, not yet built.
