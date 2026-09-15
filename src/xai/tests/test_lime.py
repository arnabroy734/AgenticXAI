"""
test_lime.py

Smoke test for lime.py using two small (100-point), hand-crafted
datasets with a known, sensible ground-truth relationship - one
regression, one classification - each with a numeric AND a categorical
feature. Trains a simple model on each, wraps it in a predict_fn, runs
LIME on a chosen test instance, and saves the result as JSON + a bar
chart plot.

Datasets (deliberately designed to make common sense, so LIME's output
signs can be sanity-checked against intuition):

  Regression - salary prediction:
      years_experience (numeric) -> more experience, higher salary
      education (categorical: bachelors/masters/phd) -> phd pays more
      department (categorical: engineering/sales) -> engineering pays more

  Classification - loan approval (ties into the bank use case):
      income (numeric) -> higher income, more likely approved
      employment_status (categorical) -> unemployed hurts approval
      existing_loan (categorical: yes/no) -> having one hurts approval

Usage:
    python test_lime.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from xai import lime as lime_module

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RANDOM_SEED = 42
N_POINTS = 100


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def generate_regression_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    years_experience = rng.uniform(0, 20, N_POINTS)
    education = rng.choice(["bachelors", "masters", "phd"], N_POINTS, p=[0.5, 0.35, 0.15])
    department = rng.choice(["engineering", "sales"], N_POINTS, p=[0.6, 0.4])

    education_bonus = {"bachelors": 0, "masters": 8000, "phd": 20000}
    department_bonus = {"engineering": 10000, "sales": 0}

    salary = (
        40000
        + years_experience * 3000
        + np.array([education_bonus[e] for e in education])
        + np.array([department_bonus[d] for d in department])
        + rng.normal(0, 5000, N_POINTS)
    )

    return pd.DataFrame({
        "years_experience": years_experience,
        "education": education,
        "department": department,
        "salary": salary,
    })


def generate_classification_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    income = rng.uniform(20000, 150000, N_POINTS)
    employment_status = rng.choice(["employed", "unemployed", "self-employed"], N_POINTS, p=[0.6, 0.15, 0.25])
    existing_loan = rng.choice(["yes", "no"], N_POINTS, p=[0.3, 0.7])

    employment_effect = {"employed": 1.0, "unemployed": -2.0, "self-employed": 0.0}
    existing_loan_effect = {"yes": -1.5, "no": 0.0}

    logit = (
        -3.0
        + income / 30000.0
        + np.array([employment_effect[e] for e in employment_status])
        + np.array([existing_loan_effect[e] for e in existing_loan])
        + rng.normal(0, 0.5, N_POINTS)
    )
    prob_approved = 1 / (1 + np.exp(-logit))
    loan_approved = rng.binomial(1, prob_approved)

    return pd.DataFrame({
        "income": income,
        "employment_status": employment_status,
        "existing_loan": existing_loan,
        "loan_approved": loan_approved,
    })


# ---------------------------------------------------------------------
# Feature stats (mirrors checkpoints/<model>/feature_stats.json shape)
# ---------------------------------------------------------------------

def compute_feature_stats(df, numeric_cols, categorical_cols):
    numeric_features = {}
    for col in numeric_cols:
        values = df[col]
        numeric_features[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
        }

    categorical_features = {}
    for col in categorical_cols:
        counts = df[col].value_counts()
        proportions = (counts / counts.sum()).round(4)
        categorical_features[col] = {
            "categories": counts.index.tolist(),
            "frequencies": proportions.to_dict(),
        }

    return {"numeric_features": numeric_features, "categorical_features": categorical_features}


# ---------------------------------------------------------------------
# Model training + predict_fn wrapping
# ---------------------------------------------------------------------

def encode_row(raw_dict, numeric_cols, categorical_features, model_columns):
    row = {col: raw_dict[col] for col in numeric_cols}
    for col, stats in categorical_features.items():
        for category in stats["categories"]:
            row[f"{col}_{category}"] = 1 if raw_dict[col] == category else 0
    return np.array([row[c] for c in model_columns], dtype=float).reshape(1, -1)


def train_regression_model(df, numeric_cols, categorical_cols, target_col):
    feature_stats = compute_feature_stats(df, numeric_cols, categorical_cols)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    model_columns = [c for c in df_encoded.columns if c != target_col]

    X = df_encoded[model_columns].values.astype(float)
    y = df_encoded[target_col].values.astype(float)

    model = LinearRegression()
    model.fit(X, y)

    def predict_fn(raw_dict):
        vec = encode_row(raw_dict, numeric_cols, feature_stats["categorical_features"], model_columns)
        return float(model.predict(vec)[0])

    return model, feature_stats, predict_fn


def train_classification_model(df, numeric_cols, categorical_cols, target_col):
    feature_stats = compute_feature_stats(df, numeric_cols, categorical_cols)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    model_columns = [c for c in df_encoded.columns if c != target_col]

    X = df_encoded[model_columns].values.astype(float)
    y = df_encoded[target_col].values.astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    def predict_fn(raw_dict):
        # LIME explains the predicted probability of the positive class (approved=1)
        vec = encode_row(raw_dict, numeric_cols, feature_stats["categorical_features"], model_columns)
        return float(model.predict_proba(vec)[0][1])

    return model, feature_stats, predict_fn


# ---------------------------------------------------------------------
# Reporting: JSON + plot
# ---------------------------------------------------------------------

def save_json(result, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON -> {path}")


def save_plot(result, title, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    attributions = result["attributions"]
    names = list(attributions.keys())
    values = list(attributions.values())

    order = np.argsort(np.abs(values))
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in values]

    plt.figure(figsize=(7, 4))
    plt.barh(names, values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Local attribution (LIME coefficient)")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot -> {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_regression_test():
    print("\n=== Regression: salary prediction ===")
    df = generate_regression_dataset()
    numeric_cols = ["years_experience"]
    categorical_cols = ["education", "department"]

    model, feature_stats, predict_fn = train_regression_model(df, numeric_cols, categorical_cols, "salary")

    # A senior, PhD-holding engineer - expect strongly positive attributions
    # for years_experience, education=phd, and department=engineering.
    test_instance = {
        "years_experience": 15.0,
        "education": "phd",
        "department": "engineering",
    }

    result = lime_module.explain_instance(
        x=test_instance,
        predict_fn=predict_fn,
        feature_stats=feature_stats,
        num_samples=500,
        random_state=RANDOM_SEED,
    )

    print(f"Instance: {test_instance}")
    print(f"Predicted salary: {result['prediction']:.2f}")
    print(f"Attributions: {result['attributions']}")
    print(f"Local surrogate R^2: {result['local_model_r2']:.4f}")

    save_json(result, "lime_regression_result.json")
    save_plot(result, "LIME - Salary prediction (senior PhD engineer)", "lime_regression_plot.png")

    return result


def run_classification_test():
    print("\n=== Classification: loan approval ===")
    df = generate_classification_dataset()
    numeric_cols = ["income"]
    categorical_cols = ["employment_status", "existing_loan"]

    model, feature_stats, predict_fn = train_classification_model(df, numeric_cols, categorical_cols, "loan_approved")

    # High income, employed, no existing loan - expect positive attributions
    # for income and employment_status=employed, and existing_loan=no should
    # not drag the prediction down.
    test_instance = {
        "income": 95000.0,
        "employment_status": "employed",
        "existing_loan": "no",
    }

    result = lime_module.explain_instance(
        x=test_instance,
        predict_fn=predict_fn,
        feature_stats=feature_stats,
        num_samples=500,
        random_state=RANDOM_SEED,
    )

    print(f"Instance: {test_instance}")
    print(f"Predicted approval probability: {result['prediction']:.4f}")
    print(f"Attributions: {result['attributions']}")
    print(f"Local surrogate R^2: {result['local_model_r2']:.4f}")

    save_json(result, "lime_classification_result.json")
    save_plot(result, "LIME - Loan approval probability (employed, no existing loan)", "lime_classification_plot.png")

    return result


if __name__ == "__main__":
    run_regression_test()
    run_classification_test()
    print("\nAll LIME tests completed.")