"""
bbc_news_train.py

Splits the BBC News dataset and trains a classical TF-IDF + Logistic
Regression text classifier. Saves the splits (raw text, unvectorized)
so bbc_news_encoder_train.py can reuse the exact same train/val/test
split without needing to re-split the data.

Usage:
    python bbc_news_train.py
"""

import os
import json
import glob

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "bbc_news")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "bbc_news_tfidf")

TEXT_COL = "Text"
TARGET_COL = "Category"

# Grid search range for Logistic Regression hyperparameter tuning
C_GRID = [0.01, 0.1, 1.0, 10.0]
MAX_FEATURES = 20000


def load_raw_data():
    csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {DATASET_DIR}")
    df = pd.read_csv(csv_files[0])
    print(f"Loaded raw data: {df.shape} from {csv_files[0]}")
    return df


def drop_missing(df):
    before = len(df)
    df = df.dropna(subset=[TEXT_COL, TARGET_COL]).reset_index(drop=True)
    print(f"Dropped missing values: {before} -> {len(df)} rows")
    return df


def split_data(df):
    # 70/10/20 train/val/test, stratified on the target to preserve
    # category balance across splits.
    train_val, test = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=RANDOM_STATE, stratify=train_val[TARGET_COL]
    )
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train_df, val_df, test_df):
    # Saved as raw text (unvectorized) - the encoder script needs raw
    # text too, and vectorization is specific to the classical model.
    train_df[[TEXT_COL, TARGET_COL]].to_csv(os.path.join(DATASET_DIR, "train.csv"), index=False)
    val_df[[TEXT_COL, TARGET_COL]].to_csv(os.path.join(DATASET_DIR, "val.csv"), index=False)
    test_df[[TEXT_COL, TARGET_COL]].to_csv(os.path.join(DATASET_DIR, "test.csv"), index=False)
    print("Saved train.csv, val.csv, test.csv (raw text, unvectorized)")


def save_label_stats(train_df):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    counts = train_df[TARGET_COL].value_counts()
    proportions = (counts / counts.sum()).round(4)
    label_stats = {
        "categories": counts.index.tolist(),
        "frequencies": proportions.to_dict(),
    }
    path = os.path.join(CHECKPOINT_DIR, "label_stats.json")
    with open(path, "w") as f:
        json.dump(label_stats, f, indent=2)
    print(f"Saved label stats -> {path}")
    return label_stats


def tune_tfidf_logreg(train_df, val_df):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    X_train = vectorizer.fit_transform(train_df[TEXT_COL])
    X_val = vectorizer.transform(val_df[TEXT_COL])
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    best_val_f1 = -1.0
    best_params = None
    best_model = None
    grid_results = []

    for C in C_GRID:
        model = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        val_acc = accuracy_score(y_val, val_pred)

        grid_results.append({"C": C, "val_f1_macro": float(val_f1), "val_accuracy": float(val_acc)})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = {"C": C}
            best_model = model

    print(f"Best hyperparameters: {best_params} (val macro-F1: {best_val_f1:.4f})")
    return vectorizer, best_model, best_params, best_val_f1, grid_results


def evaluate_on_test(vectorizer, model, test_df):
    X_test = vectorizer.transform(test_df[TEXT_COL])
    y_test = test_df[TARGET_COL].values
    test_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, test_pred)
    f1_macro = f1_score(y_test, test_pred, average="macro")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 (macro): {f1_macro:.4f}")
    return accuracy, f1_macro


def save_model_and_metrics(vectorizer, model, best_params, best_val_f1, grid_results, accuracy, f1_macro):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(CHECKPOINT_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(CHECKPOINT_DIR, "vectorizer.pkl"))
    print(f"Saved model + vectorizer -> {CHECKPOINT_DIR}")

    metrics = {
        "best_hyperparameters": best_params,
        "val_f1_macro_at_best": best_val_f1,
        "test_accuracy": accuracy,
        "test_f1_macro": f1_macro,
        "grid_search_results": grid_results,
    }
    with open(os.path.join(CHECKPOINT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {os.path.join(CHECKPOINT_DIR, 'metrics.json')}")


def main():
    df = load_raw_data()
    df = drop_missing(df)

    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df)
    save_label_stats(train_df)

    vectorizer, best_model, best_params, best_val_f1, grid_results = tune_tfidf_logreg(train_df, val_df)
    accuracy, f1_macro = evaluate_on_test(vectorizer, best_model, test_df)

    save_model_and_metrics(vectorizer, best_model, best_params, best_val_f1, grid_results, accuracy, f1_macro)

    print("\nDone.")


if __name__ == "__main__":
    main()