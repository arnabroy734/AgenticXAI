"""
house_prices_train.py

Preprocesses the House Prices dataset, splits it, standardizes numeric
features and target, saves feature statistics, and trains an ElasticNet
(L1 + L2) regression model with hyperparameter tuning on the val split.

Usage:
    python house_prices_train.py
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "house_prices")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "house_prices_linear")

TARGET_COL = "price"
NUMERIC_COLS = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_COLS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus",
]

# Grid search ranges for ElasticNet hyperparameter tuning
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


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


def split_data(df):
    # 70/10/20 train/val/test. First carve out test (20%), then split
    # the remainder into train (70% of total) and val (10% of total).
    train_val, test = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=0.125, random_state=RANDOM_STATE)
    # 0.125 of the remaining 80% = 10% of total; leaves 70% for train.
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
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


def standardize_numeric_and_target(train_df, val_df, test_df, numeric_stats):
    target_mean = float(train_df[TARGET_COL].mean())
    target_std = float(train_df[TARGET_COL].std())

    out = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df = df.copy()
        df[f"{TARGET_COL}_raw"] = df[TARGET_COL]
        df[TARGET_COL] = (df[TARGET_COL] - target_mean) / target_std

        for col in NUMERIC_COLS:
            mean = numeric_stats[col]["mean"]
            std = numeric_stats[col]["std"]
            df[col] = (df[col] - mean) / std

        out[name] = df

    target_stats = {"mean": target_mean, "std": target_std}
    return out["train"], out["val"], out["test"], target_stats


def save_splits(train_df, val_df, test_df):
    train_df.to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATASET_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)
    print("Saved train.csv, val.csv, test.csv")


def save_feature_stats(numeric_stats, categorical_stats, target_stats, model_feature_columns):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    feature_stats = {
        "numeric_features": numeric_stats,
        "categorical_features": categorical_stats,
        "target": target_stats,
        "model_feature_columns": model_feature_columns,
    }
    path = os.path.join(CHECKPOINT_DIR, "feature_stats.json")
    with open(path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"Saved feature stats -> {path}")


def get_xy(df, feature_columns):
    X = df[feature_columns].values.astype(float)
    y = df[TARGET_COL].values.astype(float)
    return X, y


def tune_elastic_net(train_df, val_df, feature_columns):
    X_train, y_train = get_xy(train_df, feature_columns)
    X_val, y_val = get_xy(val_df, feature_columns)

    best_val_mse = float("inf")
    best_params = None
    best_model = None
    grid_results = []

    for alpha in ALPHA_GRID:
        for l1_ratio in L1_RATIO_GRID:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE, max_iter=10000)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)

            grid_results.append({"alpha": alpha, "l1_ratio": l1_ratio, "val_mse": float(val_mse)})

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_params = {"alpha": alpha, "l1_ratio": l1_ratio}
                best_model = model

    print(f"Best hyperparameters: {best_params} (val MSE: {best_val_mse:.4f})")
    return best_model, best_params, best_val_mse, grid_results


def evaluate_on_test(model, test_df, feature_columns, target_stats):
    X_test, y_test_standardized = get_xy(test_df, feature_columns)
    y_test_raw = test_df[f"{TARGET_COL}_raw"].values.astype(float)

    pred_standardized = model.predict(X_test)
    pred_raw = pred_standardized * target_stats["std"] + target_stats["mean"]

    mse_raw = mean_squared_error(y_test_raw, pred_raw)
    r2_raw = r2_score(y_test_raw, pred_raw)

    print(f"Test MSE (raw price scale): {mse_raw:.2f}")
    print(f"Test R2: {r2_raw:.4f}")
    return mse_raw, r2_raw


def save_model_and_metrics(model, best_params, best_val_mse, grid_results, mse_raw, r2_raw):
    model_path = os.path.join(CHECKPOINT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")

    metrics = {
        "best_hyperparameters": best_params,
        "val_mse_at_best": best_val_mse,
        "test_mse_raw_scale": mse_raw,
        "test_r2": r2_raw,
        "grid_search_results": grid_results,
    }
    metrics_path = os.path.join(CHECKPOINT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


def main():
    df = load_raw_data()
    df = drop_missing(df)

    train_df, val_df, test_df = split_data(df)

    numeric_stats = compute_numeric_stats(train_df)
    categorical_stats = compute_categorical_stats(train_df)

    train_df, val_df, test_df = one_hot_encode(train_df, val_df, test_df)

    train_df, val_df, test_df, target_stats = standardize_numeric_and_target(
        train_df, val_df, test_df, numeric_stats
    )

    save_splits(train_df, val_df, test_df)

    feature_columns = [c for c in train_df.columns if c not in (TARGET_COL, f"{TARGET_COL}_raw")]
    save_feature_stats(numeric_stats, categorical_stats, target_stats, feature_columns)

    best_model, best_params, best_val_mse, grid_results = tune_elastic_net(train_df, val_df, feature_columns)
    mse_raw, r2_raw = evaluate_on_test(best_model, test_df, feature_columns, target_stats)

    save_model_and_metrics(best_model, best_params, best_val_mse, grid_results, mse_raw, r2_raw)

    print("\nDone.")


if __name__ == "__main__":
    main()