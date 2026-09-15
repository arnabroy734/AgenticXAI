# AgenticXAI

An agentic framework for black-box explainability across classical ML and generative models. The end goal: an agent that reads a model's API description, infers what kind of model it is, generates synthetic data where needed, runs the right explainability techniques (LIME, PDP, Permutation Importance for tabular models; leave-one-out occlusion and self-explanation prompting for LLMs), validates the results, and produces a report - without needing white-box access to the model.

This project is motivated by the RBI's *Guidance on Regulatory Principles for Model Risk Management, 2026*, which requires explainability and risk assessment for models used by regulated financial entities, including third-party/vendor models where internal access is typically unavailable.

## Project structure

```
.
├── demo_models/              # Dataset + model building, for demo purposes only
│   ├── datasets/              # Downloaded datasets (house_prices, bank_marketing, bbc_news)
│   ├── checkpoints/            # Trained model artifacts, metrics, feature stats
│   ├── download_datasets.py    # Pulls all 3 datasets via curl
│   ├── house_prices_train.py   # Preprocessing + ElasticNet training (regression)
│   ├── house_prices_rf_train.py       # Random Forest (evaluated, not used - see Results)
│   ├── house_prices_gb_train.py       # Gradient Boosting (evaluated, not used - see Results)
│   ├── house_prices_poly_train.py     # Polynomial features + ElasticNet (evaluated, not used)
│   ├── house_prices_ann_train.py      # 2-layer ANN (kept alongside ElasticNet)
│   ├── bank_marketing_train.py        # Classification model (in progress)
│   └── bbc_news_train.py              # Text classification, fine-tuned encoder (in progress)
│
├── demo_serving/              # Model serving API
│   ├── app.py                  # FastAPI app: predict + describe endpoints per model
│   ├── run_server.sh           # Starts the server
│   └── test.py                 # Hits the running server with sample data points
│
├── src/                        # Core project logic (explainability tools, agent) - to be built
│
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Task | Features | Status |
|---|---|---|---|
| House Prices (`yasserh/housing-prices-dataset`) | Regression | Numeric + categorical | Done |
| Bank Marketing (`janiobachmann/bank-marketing-dataset`) | Classification | Numeric + categorical | In progress |
| BBC News (GitHub CSV) | Text classification | Raw text | In progress |

## Preprocessing pipeline (House Prices)

1. Drop missing values.
2. One-hot encode categorical features (all categories kept, fit on train split only to avoid leakage).
3. Split 70/10/20 (train/val/test), val reserved for hyperparameter tuning.
4. Standardize numeric features and the target using train-split mean/std.
5. Save `train.csv`, `val.csv`, `test.csv` (standardized features + one-hot columns + both standardized and raw target).
6. Save feature statistics to `checkpoints/house_prices_linear/feature_stats.json`: per numeric feature (mean, std, min, max, p25, p75), per categorical feature (category list + frequencies), and target stats - needed later for the agent's synthetic data generation and for inverse-transforming predictions.

Every other model script (RF, GB, polynomial+ElasticNet, ANN) reuses these exact splits and feature stats with no re-encoding, and each guards against being run before `house_prices_train.py` with a clear error message.

## Results (House Prices)

| Model | Test R² | Notes |
|---|---|---|
| ElasticNet (L1+L2) | **0.679** | Final choice |
| 2-layer ANN | 0.65 | Kept alongside ElasticNet |
| Random Forest | lower | Evaluated, dropped |
| Gradient Boosting | lower | Evaluated, dropped |
| Polynomial + ElasticNet | lower | Evaluated, dropped |

At this dataset size (~545 rows), more flexible models consistently failed to beat a well-regularized linear model - consistent with published benchmarks showing the gap between tree ensembles/deep models and linear baselines shrinks (and can invert) on small tabular datasets. ElasticNet and the ANN were kept as the two models taken forward into serving and explainability.

## Serving API

A single FastAPI app serves both House Price models. Each model exposes two endpoints:

- `POST /house_prices/{model_key}/predict` - takes raw feature values (not preprocessed), returns a prediction in original units.
- `GET /house_prices/{model_key}/describe` - returns job type, per-feature schema (numeric stats or categorical categories/frequencies, target excluded), and an example request body.

`model_key` is `linear` or `ann`.

### Running it

```bash
cd demo_serving
./run_server.sh
```

In a separate terminal:

```bash
python test.py
```

`test.py` sends two sample data points to both models' `/predict` endpoints, checks both `/describe` endpoints, and confirms an invalid categorical value is correctly rejected with a 400.

## Status

- [x] House Prices: preprocessing, model comparison, final models (ElasticNet + ANN), serving API
- [ ] Bank Marketing: preprocessing + classification model
- [ ] BBC News: fine-tuned encoder text classification model
- [ ] `src/`: explainability tools (LIME, PDP, Permutation Importance, leave-one-out occlusion, self-explanation prompting)
- [ ] Agent: reads API description, infers model type, generates synthetic data, runs tools, validates, produces report