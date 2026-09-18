"""
bbc_news_encoder_train.py

Fine-tunes a lightweight pretrained encoder (DistilBERT) on the BBC
News dataset, reusing the train/val/test splits and label list already
produced by bbc_news_train.py. No re-splitting is done here.

Requires internet access to download the pretrained model/tokenizer
from Hugging Face on first run.

Usage:
    python bbc_news_encoder_train.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

RANDOM_STATE = 42
MODEL_NAME = "distilbert-base-uncased"  # lightweight pretrained encoder

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "bbc_news")
TFIDF_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "bbc_news_tfidf")
ENCODER_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "bbc_news_encoder")

TEXT_COL = "Text"
TARGET_COL = "Category"

NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 256


def check_splits_exist():
    required_files = ["train.csv", "val.csv", "test.csv"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(DATASET_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing split file(s) {missing} in {DATASET_DIR}. "
            "Run bbc_news_train.py first to generate the train/val/test splits."
        )

    label_stats_path = os.path.join(TFIDF_CHECKPOINT_DIR, "label_stats.json")
    if not os.path.exists(label_stats_path):
        raise FileNotFoundError(
            f"Missing label_stats.json at {label_stats_path}. "
            "Run bbc_news_train.py first to generate label stats."
        )


def load_splits_and_labels():
    train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATASET_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

    with open(os.path.join(TFIDF_CHECKPOINT_DIR, "label_stats.json")) as f:
        label_stats = json.load(f)

    categories = sorted(label_stats["categories"])  # fixed, deterministic label order
    label_to_id = {c: i for i, c in enumerate(categories)}
    id_to_label = {i: c for c, i in label_to_id.items()}

    print(f"Loaded splits -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")
    print(f"Categories: {categories}")
    return train_df, val_df, test_df, label_to_id, id_to_label


def to_hf_dataset(df, tokenizer, label_to_id):
    df = df.copy()
    df["label"] = df[TARGET_COL].map(label_to_id)
    ds = Dataset.from_pandas(df[[TEXT_COL, "label"]])

    def tokenize(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns([TEXT_COL])
    ds.set_format("torch")
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": accuracy, "f1_macro": f1_macro}


def main():
    check_splits_exist()
    train_df, val_df, test_df, label_to_id, id_to_label = load_splits_and_labels()

    print(f"\nLoading pretrained tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_to_id), id2label=id_to_label, label2id=label_to_id
    )

    train_ds = to_hf_dataset(train_df, tokenizer, label_to_id)
    val_ds = to_hf_dataset(val_df, tokenizer, label_to_id)
    test_ds = to_hf_dataset(test_df, tokenizer, label_to_id)

    os.makedirs(ENCODER_CHECKPOINT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(ENCODER_CHECKPOINT_DIR, "training_run"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=RANDOM_STATE,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\nFine-tuning...")
    trainer.train()

    print("\nEvaluating on val split...")
    val_metrics = trainer.evaluate(val_ds)
    print(f"Val metrics: {val_metrics}")

    print("\nEvaluating on test split...")
    test_metrics = trainer.evaluate(test_ds)
    print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics['eval_f1_macro']:.4f}")

    print(f"\nSaving model + tokenizer -> {ENCODER_CHECKPOINT_DIR}")
    model.save_pretrained(ENCODER_CHECKPOINT_DIR)
    tokenizer.save_pretrained(ENCODER_CHECKPOINT_DIR)

    metrics = {
        "model_name": MODEL_NAME,
        "num_epochs": NUM_EPOCHS,
        "val_accuracy": val_metrics["eval_accuracy"],
        "val_f1_macro": val_metrics["eval_f1_macro"],
        "test_accuracy": test_metrics["eval_accuracy"],
        "test_f1_macro": test_metrics["eval_f1_macro"],
        "label_to_id": label_to_id,
    }
    with open(os.path.join(ENCODER_CHECKPOINT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {os.path.join(ENCODER_CHECKPOINT_DIR, 'metrics.json')}")

    print("\nDone.")


if __name__ == "__main__":
    main()