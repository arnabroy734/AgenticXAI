"""
house_pred_lime_pdp.py

Runs LIME and PDP against the House Price serving API for BOTH models
(linear = ElasticNet, ann = 2-layer ANN). Calls predict_fn over HTTP
against a running server (see demo_serving/run_server.sh), fetches
feature_stats from the server's /describe endpoint (no need to load
checkpoints directly), and uses the provided val.csv as the background
dataset for PDP - reconstructed back to raw feature values, since
val.csv is the preprocessed (standardized + one-hot) split.

For each model:
    1. Runs LIME on one hardcoded test instance (the first row of the
       reconstructed val set).
    2. Picks the top 2 numeric and top 2 categorical features by
       |LIME attribution| for that instance.
    3. Runs PDP on those 4 features, using the full reconstructed val
       set as the background data.

All results (JSON) and plots are saved in raw/original scale - not
standardized - since predict_fn already operates in raw feature space
end to end.

Usage:
    1. Start the server first:
        cd demo_serving && ./run_server.sh
    2. In another terminal:
        python house_pred_lime_pdp.py
"""

import os
import json

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from xai import lime as lime_module
from xai import pdp as pdp_module

BASE_URL = "http://localhost:8000"
MODEL_KEYS = ["linear", "ann"]
TOP_N_PER_TYPE = 2

# These fields are strictly integer-typed in the serving API's schema
# (real counts: bedrooms, bathrooms, stories, parking). LIME/PDP perturb
# numeric features with continuous values, so these must be rounded back
# to whole numbers before each API call - only "area" is genuinely
# continuous in this schema.
INTEGER_FIELDS = {"bedrooms", "bathrooms", "stories", "parking"}

VAL_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "demo_models", "datasets", "house_prices", "val.csv"
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def check_server_running():
    try:
        requests.get(f"{BASE_URL}/house_prices/linear/describe", timeout=3)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not connect to the server at {BASE_URL}. "
            f"Start it first with: cd demo_serving && ./run_server.sh"
        ) from e


def fetch_feature_stats(model_key):
    r = requests.get(f"{BASE_URL}/house_prices/{model_key}/describe")
    r.raise_for_status()
    body = r.json()

    numeric_features, categorical_features = {}, {}
    for f in body["features"]:
        if f["field_type"] == "numeric":
            numeric_features[f["name"]] = f["stats"]
        else:
            categorical_features[f["name"]] = {
                "categories": f["categories"],
                "frequencies": f["frequencies"],
            }

    return {"numeric_features": numeric_features, "categorical_features": categorical_features}


def sanitize_for_api(raw_dict):
    """Rounds integer-typed fields back to whole numbers before sending
    to the API, since LIME/PDP perturb all numeric fields continuously."""
    sanitized = dict(raw_dict)
    for field in INTEGER_FIELDS:
        if field in sanitized:
            sanitized[field] = int(round(sanitized[field]))
    return sanitized


def make_predict_fn(model_key):
    def predict_fn(raw_dict):
        r = requests.post(f"{BASE_URL}/house_prices/{model_key}/predict", json=sanitize_for_api(raw_dict))
        r.raise_for_status()
        return float(r.json()["predicted_price"])
    return predict_fn


def reconstruct_raw_row(row, feature_stats):
    """
    val.csv rows are preprocessed (standardized numeric + one-hot
    categorical). This inverts both back to raw values, using the same
    mean/std the server reports (computed on the raw scale at training
    time) and by finding which one-hot dummy is set per categorical group.
    """
    raw = {}

    for name, stats in feature_stats["numeric_features"].items():
        standardized_value = row[name]
        raw[name] = standardized_value * stats["std"] + stats["mean"]

    for name, stats in feature_stats["categorical_features"].items():
        matched = None
        for category in stats["categories"]:
            col = f"{name}_{category}"
            if row.get(col) in (True, 1, "True", "1", 1.0):
                matched = category
                break
        if matched is None:
            raise ValueError(f"Could not reconstruct categorical value for '{name}' from row: {row}")
        raw[name] = matched

    return raw


def load_background_data(feature_stats):
    df = pd.read_csv(VAL_CSV_PATH)
    rows = df.to_dict("records")
    return [reconstruct_raw_row(row, feature_stats) for row in rows]


def pick_top_features(attributions, feature_stats, top_n=2):
    numeric_names = set(feature_stats["numeric_features"].keys())
    categorical_names = set(feature_stats["categorical_features"].keys())

    numeric_attrs = {k: v for k, v in attributions.items() if k in numeric_names}
    categorical_attrs = {k: v for k, v in attributions.items() if k in categorical_names}

    top_numeric = sorted(numeric_attrs, key=lambda k: abs(numeric_attrs[k]), reverse=True)[:top_n]
    top_categorical = sorted(categorical_attrs, key=lambda k: abs(categorical_attrs[k]), reverse=True)[:top_n]

    return top_numeric, top_categorical


def save_json(result, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON -> {path}")


def save_lime_plot(result, title, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    attributions = result["attributions"]
    names = list(attributions.keys())
    values = list(attributions.values())

    order = np.argsort(np.abs(values))
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in values]

    plt.figure(figsize=(7, 5))
    plt.barh(names, values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Local attribution (price impact, LIME coefficient)")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot -> {path}")


def save_pdp_plot(result, title, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grid_values = result["grid_values"]
    avg_preds = result["average_predictions"]

    plt.figure(figsize=(6, 4))
    if result["feature_type"] == "numeric":
        plt.plot(grid_values, avg_preds, marker="o", color="#185FA5")
    else:
        plt.bar([str(v) for v in grid_values], avg_preds, color="#7F77DD")

    plt.xlabel(result["feature_name"])
    plt.ylabel("Average predicted price (original scale)")
    plt.title(title)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot -> {path}")


def run_for_model(model_key, test_instance, background_data, feature_stats):
    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"{'='*60}")

    predict_fn = make_predict_fn(model_key)

    # --- LIME on the hardcoded test instance ---
    lime_result = lime_module.explain_instance(
        x=test_instance,
        predict_fn=predict_fn,
        feature_stats=feature_stats,
        num_samples=500,
        random_state=42,
    )

    print(f"\nInstance (raw scale): {test_instance}")
    print(f"Predicted price: {lime_result['prediction']:.2f}")
    print(f"Attributions: {lime_result['attributions']}")
    print(f"Local surrogate R^2: {lime_result['local_model_r2']:.4f}")

    save_json(lime_result, f"house_{model_key}_lime_result.json")
    save_lime_plot(lime_result, f"LIME - House price ({model_key})", f"house_{model_key}_lime_plot.png")

    # --- Pick top 2 numeric + top 2 categorical features from this LIME result ---
    top_numeric, top_categorical = pick_top_features(lime_result["attributions"], feature_stats, TOP_N_PER_TYPE)
    top_features = top_numeric + top_categorical
    print(f"\nTop {TOP_N_PER_TYPE} numeric features: {top_numeric}")
    print(f"Top {TOP_N_PER_TYPE} categorical features: {top_categorical}")

    # --- PDP on those 4 selected features ---
    for feature_name in top_features:
        pdp_result = pdp_module.compute_pdp(
            feature_name=feature_name,
            predict_fn=predict_fn,
            feature_stats=feature_stats,
            background_data=background_data,
            grid_size=20,
        )

        print(f"\nPDP feature: {feature_name} ({pdp_result['feature_type']})")
        print(f"Grid values: {pdp_result['grid_values']}")
        print(f"Average predictions: {pdp_result['average_predictions']}")

        save_json(pdp_result, f"house_{model_key}_pdp_{feature_name}.json")
        save_pdp_plot(
            pdp_result,
            f"PDP - House price vs {feature_name} ({model_key})",
            f"house_{model_key}_pdp_{feature_name}.png",
        )


def main():
    check_server_running()

    # feature_stats is identical across both checkpoints (same splits/preprocessing),
    # but fetched per-model below anyway to stay self-contained per model.
    reconstruction_stats = fetch_feature_stats("linear")
    background_data = load_background_data(reconstruction_stats)
    print(f"Loaded and reconstructed background data: {len(background_data)} raw rows from val.csv")

    test_instance = background_data[0]

    for model_key in MODEL_KEYS:
        feature_stats = fetch_feature_stats(model_key)
        run_for_model(model_key, test_instance, background_data, feature_stats)

    print("\nAll house price LIME + PDP tests completed.")


if __name__ == "__main__":
    main()