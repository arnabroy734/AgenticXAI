"""
bank_marketing_train.py

Preprocesses the Bank Marketing dataset, splits it, standardizes
numeric features, saves feature statistics, and trains a Logistic
Regression classifier with hyperparameter tuning on the val split.

Two data-quality decisions applied before training:
    1. 'duration' is dropped. It is a well-known leakage feature in this
       dataset - call duration correlates almost mechanically with the
       target (you don't get a long call with someone who declines
       early), so leaving it in produces an artificially strong, and
       misleading, model.
    2. 'pdays' uses -1 as a sentinel meaning "never contacted before",
       not a real numeric distance. A derived binary flag
       'previously_contacted' is added before pdays is treated as an
       ordinary numeric feature, so -1 isn't perturbed downstream as if
       it were just another point on a continuous scale.

Usage:
    python bank_marketing_train.py
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "bank_marketing")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "bank_marketing")

TARGET_COL = "deposit"
POSITIVE_CLASS = "yes"

LEAKAGE_COLS = ["duration"]

NUMERIC_COLS = ["age", "balance", "day", "campaign", "pdays", "previous", "previously_contacted"]
CATEGORICAL_COLS = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

# Grid search range for Logistic Regression hyperparameter tuning
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def load_raw_data():
    csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {DATASET_DIR}")
    df = pd.read_csv(csv_files[0])
    print(f"Loaded raw data: {df.shape} from {csv_files[0]}")
    return df


def drop_missing(df):
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped missing values: {before} -> {len(df)} rows")
    return df


def apply_data_quality_fixes(df):
    df = df.drop(columns=LEAKAGE_COLS)
    print(f"Dropped leakage column(s): {LEAKAGE_COLS}")

    df["previously_contacted"] = (df["pdays"] != -1).astype(int)
    print("Added derived column: previously_contacted (1 if pdays != -1, else 0)")

    return df


def encode_target(df):
    df[TARGET_COL] = (df[TARGET_COL] == POSITIVE_CLASS).astype(int)
    return df


def split_data(df):
    # 70/10/20 train/val/test, stratified on the target to preserve class
    # balance across splits (important for classification, unlike regression).
    train_val, test = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=RANDOM_STATE, stratify=train_val[TARGET_COL]
    )
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"Positive class rate -> train: {train[TARGET_COL].mean():.3f}, "
          f"val: {val[TARGET_COL].mean():.3f}, test: {test[TARGET_COL].mean():.3f}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def compute_numeric_stats(train_df):
    stats = {}
    for col in NUMERIC_COLS:
        values = train_df[col]
        stats[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
        }
    return stats


def compute_categorical_stats(train_df):
    stats = {}
    for col in CATEGORICAL_COLS:
        counts = train_df[col].value_counts()
        proportions = (counts / counts.sum()).round(4)
        stats[col] = {
            "categories": counts.index.tolist(),
            "frequencies": proportions.to_dict(),
        }
    return stats


def one_hot_encode(train_df, val_df, test_df):
    # Fix categories from the train split only, to avoid leaking
    # val/test category information into preprocessing.
    encoded = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            train_categories = sorted(train_df[col].unique())
            df[col] = pd.Categorical(df[col], categories=train_categories)
        df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
        encoded[name] = df
    return encoded["train"], encoded["val"], encoded["test"]


def standardize_numeric(train_df, val_df, test_df, numeric_stats):
    out = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df = df.copy()
        for col in NUMERIC_COLS:
            mean = numeric_stats[col]["mean"]
            std = numeric_stats[col]["std"]
            df[col] = (df[col] - mean) / std if std > 0 else 0.0
        out[name] = df
    return out["train"], out["val"], out["test"]


def save_splits(train_df, val_df, test_df):
    train_df.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATASET_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)
    print("Saved train.csv, val.csv, test.csv")


def save_feature_stats(numeric_stats, categorical_stats, model_feature_columns):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    feature_stats = {
        "numeric_features": numeric_stats,
        "categorical_features": categorical_stats,
        "target": {
            "positive_class": POSITIVE_CLASS,
            "encoding": {POSITIVE_CLASS: 1, "no": 0},
        },
        "model_feature_columns": model_feature_columns,
    }
    path = os.path.join(CHECKPOINT_DIR, "feature_stats.json")
    with open(path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"Saved feature stats -> {path}")


def get_xy(df, feature_columns):
    X = df[feature_columns].values.astype(float)
    y = df[TARGET_COL].values.astype(int)
    return X, y


def tune_logistic_regression(train_df, val_df, feature_columns):
    X_train, y_train = get_xy(train_df, feature_columns)
    X_val, y_val = get_xy(val_df, feature_columns)

    best_val_f1 = -1.0
    best_params = None
    best_model = None
    grid_results = []

    for C in C_GRID:
        model = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, val_pred)

        grid_results.append({"C": C, "val_f1": float(val_f1), "val_accuracy": float(val_acc)})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = {"C": C}
            best_model = model

    print(f"Best hyperparameters: {best_params} (val F1: {best_val_f1:.4f})")
    return best_model, best_params, best_val_f1, grid_results


def evaluate_on_test(model, test_df, feature_columns):
    X_test, y_test = get_xy(test_df, feature_columns)
    test_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1: {f1:.4f}")
    return accuracy, f1


def save_model_and_metrics(model, best_params, best_val_f1, grid_results, accuracy, f1):
    model_path = os.path.join(CHECKPOINT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")

    metrics = {
        "best_hyperparameters": best_params,
        "val_f1_at_best": best_val_f1,
        "test_accuracy": accuracy,
        "test_f1": f1,
        "grid_search_results": grid_results,
    }
    metrics_path = os.path.join(CHECKPOINT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


def main():
    df = load_raw_data()
    df = drop_missing(df)
    df = apply_data_quality_fixes(df)
    df = encode_target(df)

    train_df, val_df, test_df = split_data(df)

    numeric_stats = compute_numeric_stats(train_df)
    categorical_stats = compute_categorical_stats(train_df)

    train_df, val_df, test_df = one_hot_encode(train_df, val_df, test_df)
    train_df, val_df, test_df = standardize_numeric(train_df, val_df, test_df, numeric_stats)

    save_splits(train_df, val_df, test_df)

    feature_columns = [c for c in train_df.columns if c != TARGET_COL]
    save_feature_stats(numeric_stats, categorical_stats, feature_columns)

    best_model, best_params, best_val_f1, grid_results = tune_logistic_regression(train_df, val_df, feature_columns)
    accuracy, f1 = evaluate_on_test(best_model, test_df, feature_columns)

    save_model_and_metrics(best_model, best_params, best_val_f1, grid_results, accuracy, f1)

    print("\nDone.")


if __name__ == "__main__":
    main()