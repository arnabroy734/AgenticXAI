"""
catalog.py

Builds the "Models" tab catalog entirely by calling model_serving's
/describe for every (dataset, model_key) this demo knows about - this
service never reads checkpoint files directly (only model_serving
mounts that volume, per the deployment's volume-attachment design);
per-model test metrics are surfaced through /describe's own "metrics"
field instead (see demo_serving/app.py's _load_metrics).
"""

import os

import requests

MODEL_SERVING_URL = os.environ.get("MODEL_SERVING_URL", "http://model_serving:8000")

DATASETS = {
    "house_prices": {"display_name": "House Prices", "model_keys": ["linear", "ann"]},
    "bank_marketing": {"display_name": "Bank Marketing", "model_keys": ["logistic", "rf", "ann"]},
    "bbc_news": {"display_name": "BBC News", "model_keys": ["tfidf", "encoder"]},
}


def build_catalog() -> dict:
    catalog = {}
    for dataset_key, dataset_info in DATASETS.items():
        models = {}
        for model_key in dataset_info["model_keys"]:
            describe_url = f"{MODEL_SERVING_URL}/{dataset_key}/{model_key}/describe"
            try:
                r = requests.get(describe_url, timeout=10)
                r.raise_for_status()
                models[model_key] = r.json()
            except requests.RequestException as e:
                models[model_key] = {"error": f"Could not load this model from model_serving: {e}"}
        catalog[dataset_key] = {"display_name": dataset_info["display_name"], "models": models}
    return catalog
