"""
test_pdp.py

Smoke test for pdp.py, reusing the exact same hardcoded regression and
classification datasets/models from test_lime.py. Computes PDP for all
three features in each dataset (1 numeric + 2 categorical each) and
saves the result as JSON + a plot per feature.

Usage:
    python test_pdp.py
"""

import os
import json

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from xai import pdp as pdp_module

from test_lime import (
    generate_regression_dataset,
    generate_classification_dataset,
    train_regression_model,
    train_classification_model,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def build_background_data(df, feature_cols):
    return df[feature_cols].to_dict("records")


def save_json(result, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON -> {path}")


def save_plot(result, title, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grid_values = result["grid_values"]
    avg_preds = result["average_predictions"]

    plt.figure(figsize=(6, 4))
    if result["feature_type"] == "numeric":
        plt.plot(grid_values, avg_preds, marker="o", color="#185FA5")
    else:
        plt.bar([str(v) for v in grid_values], avg_preds, color="#7F77DD")

    plt.xlabel(result["feature_name"])
    plt.ylabel("Average predicted output")
    plt.title(title)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot -> {path}")


def run_pdp_for_all_features(df, feature_cols, feature_stats, predict_fn, label):
    background_data = build_background_data(df, feature_cols)

    for feature_name in feature_cols:
        result = pdp_module.compute_pdp(
            feature_name=feature_name,
            predict_fn=predict_fn,
            feature_stats=feature_stats,
            background_data=background_data,
            grid_size=20,
        )

        print(f"\nFeature: {feature_name} ({result['feature_type']})")
        print(f"Grid values: {result['grid_values']}")
        print(f"Average predictions: {result['average_predictions']}")

        save_json(result, f"pdp_{label}_{feature_name}.json")
        save_plot(result, f"PDP - {label} vs {feature_name}", f"pdp_{label}_{feature_name}.png")


def run_regression_pdp():
    print("\n=== PDP: salary prediction (regression) ===")
    df = generate_regression_dataset()
    numeric_cols = ["years_experience"]
    categorical_cols = ["education", "department"]
    all_cols = numeric_cols + categorical_cols

    _, feature_stats, predict_fn = train_regression_model(df, numeric_cols, categorical_cols, "salary")
    run_pdp_for_all_features(df, all_cols, feature_stats, predict_fn, "salary")


def run_classification_pdp():
    print("\n=== PDP: loan approval (classification) ===")
    df = generate_classification_dataset()
    numeric_cols = ["income"]
    categorical_cols = ["employment_status", "existing_loan"]
    all_cols = numeric_cols + categorical_cols

    _, feature_stats, predict_fn = train_classification_model(df, numeric_cols, categorical_cols, "loan_approved")
    run_pdp_for_all_features(df, all_cols, feature_stats, predict_fn, "loan_approval")


if __name__ == "__main__":
    run_regression_pdp()
    run_classification_pdp()
    print("\nAll PDP tests completed.")