"""
house_prices_ann_train.py

Trains a small 2-hidden-layer neural network (MLPRegressor) on the House
Prices dataset, reusing the train/val/test splits and feature stats
already produced by house_prices_train.py. No feature engineering is
done here - the splits are used exactly as saved.

Usage:
    python house_prices_ann_train.py
"""

import os
import json
import warnings

import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "house_prices")
LINEAR_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "house_prices_linear")
ANN_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "house_prices_ann")

TARGET_COL = "price"

# Grid search ranges. Kept small and shallow deliberately - this is a
# ~500-row dataset, a large network would just overfit.
HIDDEN_LAYER_GRID = [(8, 4), (16, 8), (32, 16)]
ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1]  # L2 regularization strength
LEARNING_RATE_INIT = 0.001
MAX_ITER = 3000


def check_splits_exist():
    required_files = ["train.csv", "val.csv", "test.csv"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(DATASET_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing split file(s) {missing} in {DATASET_DIR}. "
            "Run house_prices_train.py first to generate the train/val/test splits."
        )

    feature_stats_path = os.path.join(LINEAR_CHECKPOINT_DIR, "feature_stats.json")
    if not os.path.exists(feature_stats_path):
        raise FileNotFoundError(
            f"Missing feature_stats.json at {feature_stats_path}. "
            "Run house_prices_train.py first to generate feature stats."
        )


def load_splits():
    train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATASET_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))
    print(f"Loaded splits -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")
    return train_df, val_df, test_df


def load_feature_stats():
    path = os.path.join(LINEAR_CHECKPOINT_DIR, "feature_stats.json")
    with open(path, "r") as f:
        feature_stats = json.load(f)
    return feature_stats


def get_xy(df, feature_columns):
    X = df[feature_columns].values.astype(float)
    y = df[TARGET_COL].values.astype(float)
    return X, y


def tune_ann(train_df, val_df, feature_columns):
    X_train, y_train = get_xy(train_df, feature_columns)
    X_val, y_val = get_xy(val_df, feature_columns)

    best_val_mse = float("inf")
    best_params = None
    best_model = None
    grid_results = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for hidden_layer_sizes in HIDDEN_LAYER_GRID:
            for alpha in ALPHA_GRID:
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    learning_rate_init=LEARNING_RATE_INIT,
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                val_mse = mean_squared_error(y_val, val_pred)

                grid_results.append({
                    "hidden_layer_sizes": list(hidden_layer_sizes),
                    "alpha": alpha,
                    "val_mse": float(val_mse),
                })

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_params = {"hidden_layer_sizes": list(hidden_layer_sizes), "alpha": alpha}
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


def save_feature_stats_copy(feature_stats):
    os.makedirs(ANN_CHECKPOINT_DIR, exist_ok=True)
    dest_path = os.path.join(ANN_CHECKPOINT_DIR, "feature_stats.json")
    with open(dest_path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"Saved feature stats -> {dest_path}")


def save_model_and_metrics(model, best_params, best_val_mse, grid_results, mse_raw, r2_raw):
    os.makedirs(ANN_CHECKPOINT_DIR, exist_ok=True)

    model_path = os.path.join(ANN_CHECKPOINT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")

    metrics = {
        "best_hyperparameters": best_params,
        "val_mse_at_best": best_val_mse,
        "test_mse_raw_scale": mse_raw,
        "test_r2": r2_raw,
        "grid_search_results": grid_results,
    }
    metrics_path = os.path.join(ANN_CHECKPOINT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


def main():
    check_splits_exist()

    train_df, val_df, test_df = load_splits()
    feature_stats = load_feature_stats()
    target_stats = feature_stats["target"]

    feature_columns = feature_stats["model_feature_columns"]

    best_model, best_params, best_val_mse, grid_results = tune_ann(train_df, val_df, feature_columns)
    mse_raw, r2_raw = evaluate_on_test(best_model, test_df, feature_columns, target_stats)

    save_feature_stats_copy(feature_stats)
    save_model_and_metrics(best_model, best_params, best_val_mse, grid_results, mse_raw, r2_raw)

    print("\nDone.")


if __name__ == "__main__":
    main()