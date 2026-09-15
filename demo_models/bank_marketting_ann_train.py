"""
bank_marketing_ann_train.py

Trains a small 2-hidden-layer neural network (MLPClassifier) on the
Bank Marketing dataset, reusing the train/val/test splits and feature
stats already produced by bank_marketing_train.py. No feature
engineering is done here - the splits are used exactly as saved.

Usage:
    python bank_marketing_ann_train.py
"""

import os
import json
import warnings

import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "bank_marketing")
LOGREG_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "bank_marketing")
ANN_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "bank_marketing_ann")

TARGET_COL = "deposit"

# Grid search ranges. Kept small and shallow deliberately - a large
# network on a modest dataset would just overfit.
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
            "Run bank_marketing_train.py first to generate the train/val/test splits."
        )

    feature_stats_path = os.path.join(LOGREG_CHECKPOINT_DIR, "feature_stats.json")
    if not os.path.exists(feature_stats_path):
        raise FileNotFoundError(
            f"Missing feature_stats.json at {feature_stats_path}. "
            "Run bank_marketing_train.py first to generate feature stats."
        )


def load_splits():
    train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATASET_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))
    print(f"Loaded splits -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")
    return train_df, val_df, test_df


def load_feature_stats():
    path = os.path.join(LOGREG_CHECKPOINT_DIR, "feature_stats.json")
    with open(path, "r") as f:
        feature_stats = json.load(f)
    return feature_stats


def get_xy(df, feature_columns):
    X = df[feature_columns].values.astype(float)
    y = df[TARGET_COL].values.astype(int)
    return X, y


def tune_ann(train_df, val_df, feature_columns):
    X_train, y_train = get_xy(train_df, feature_columns)
    X_val, y_val = get_xy(val_df, feature_columns)

    best_val_f1 = -1.0
    best_params = None
    best_model = None
    grid_results = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for hidden_layer_sizes in HIDDEN_LAYER_GRID:
            for alpha in ALPHA_GRID:
                model = MLPClassifier(
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
                val_f1 = f1_score(y_val, val_pred)
                val_acc = accuracy_score(y_val, val_pred)

                grid_results.append({
                    "hidden_layer_sizes": list(hidden_layer_sizes),
                    "alpha": alpha,
                    "val_f1": float(val_f1),
                    "val_accuracy": float(val_acc),
                })

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_params = {"hidden_layer_sizes": list(hidden_layer_sizes), "alpha": alpha}
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


def save_feature_stats_copy(feature_stats):
    os.makedirs(ANN_CHECKPOINT_DIR, exist_ok=True)
    dest_path = os.path.join(ANN_CHECKPOINT_DIR, "feature_stats.json")
    with open(dest_path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"Saved feature stats -> {dest_path}")


def save_model_and_metrics(model, best_params, best_val_f1, grid_results, accuracy, f1):
    os.makedirs(ANN_CHECKPOINT_DIR, exist_ok=True)

    model_path = os.path.join(ANN_CHECKPOINT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")

    metrics = {
        "best_hyperparameters": best_params,
        "val_f1_at_best": best_val_f1,
        "test_accuracy": accuracy,
        "test_f1": f1,
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

    feature_columns = feature_stats["model_feature_columns"]

    best_model, best_params, best_val_f1, grid_results = tune_ann(train_df, val_df, feature_columns)
    accuracy, f1 = evaluate_on_test(best_model, test_df, feature_columns)

    save_feature_stats_copy(feature_stats)
    save_model_and_metrics(best_model, best_params, best_val_f1, grid_results, accuracy, f1)

    print("\nDone.")


if __name__ == "__main__":
    main()