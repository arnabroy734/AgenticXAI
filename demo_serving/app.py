"""
app.py

Serves House Price models (regression), Bank Marketing models
(classification), and BBC News text classification models behind one
FastAPI app. Each model gets two endpoints, under its own dataset prefix:

    POST /house_prices/{model_key}/predict     model_key: "linear" or "ann"
    GET  /house_prices/{model_key}/describe

    POST /bank_marketing/{model_key}/predict   model_key: "logistic", "rf", or "ann"
    GET  /bank_marketing/{model_key}/describe

    POST /bbc_news/{model_key}/predict         model_key: "tfidf" or "encoder"
    GET  /bbc_news/{model_key}/describe

The tabular datasets share generic model-loading and feature-vector-
building logic (driven entirely by feature_stats.json's shape). BBC
News is a different modality (a single raw text field, not many named
features), so it gets its own loading/prediction logic.

Every classification endpoint - binary (Bank Marketing) or multi-class
(BBC News) - returns the same shape: "predicted_class", a full
"predicted_probabilities" dict (one entry per class, not just a
positive-class scalar), and "confidence" (the probability of whichever
class was actually predicted - a display-only value). /describe's
"output" block additionally names a fixed "reference_class" - the one
class whose probability stays fixed as the explainability target
throughout LIME/PDP, since a moving target (whichever class the model
currently predicts) would produce discontinuous, misleading
explanations. For Bank Marketing this is the dataset's positive_class;
for BBC News there's no natural "positive" class, so it's declared
statically per model_key (the training-set majority class).

Usage:
    uvicorn app:app --reload
"""

import os
import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "demo_models", "checkpoints")

HOUSE_PRICE_MODELS = {
    "linear": {
        "checkpoint_dir": os.path.join(CHECKPOINTS_DIR, "house_prices_linear"),
        "display_name": "House Price - ElasticNet Linear Regression",
        "job_type": "regression",
    },
    "ann": {
        "checkpoint_dir": os.path.join(CHECKPOINTS_DIR, "house_prices_ann"),
        "display_name": "House Price - 2-layer ANN",
        "job_type": "regression",
    },
}

BANK_MARKETING_MODELS = {
    "logistic": {
        "checkpoint_dir": os.path.join(CHECKPOINTS_DIR, "bank_marketing"),
        "display_name": "Bank Marketing - Logistic Regression",
        "job_type": "classification",
    },
    "rf": {
        "checkpoint_dir": os.path.join(CHECKPOINTS_DIR, "bank_marketing_rf"),
        "display_name": "Bank Marketing - Random Forest",
        "job_type": "classification",
    },
    "ann": {
        "checkpoint_dir": os.path.join(CHECKPOINTS_DIR, "bank_marketing_ann"),
        "display_name": "Bank Marketing - 2-layer ANN",
        "job_type": "classification",
    },
}

BBC_NEWS_MODELS = {
    "tfidf": {
        "checkpoint_dir": os.path.join(CHECKPOINTS_DIR, "bbc_news_tfidf"),
        "display_name": "BBC News - TF-IDF + Logistic Regression",
        "job_type": "classification",
        "model_type": "tfidf",
        # No natural "positive class" for a 5-way problem, so the fixed
        # explainability target is declared here rather than computed at
        # request time - "sport" is the training-set majority class for
        # this dataset (23.2%, from bbc_news_tfidf/label_stats.json),
        # shared by both model_keys since they're trained on the same
        # data. Declared statically because the encoder checkpoint below
        # doesn't track label frequencies itself.
        "reference_class": "sport",
    },
    "encoder": {
        "checkpoint_dir": os.path.join(CHECKPOINTS_DIR, "bbc_news_encoder"),
        "display_name": "BBC News - Fine-tuned DistilBERT",
        "job_type": "classification",
        "model_type": "encoder",
        "reference_class": "sport",
    },
}


class HousePriceFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    prefarea: str
    furnishingstatus: str


class BankMarketingFeatures(BaseModel):
    age: int
    balance: int
    day: int
    campaign: int
    pdays: int
    previous: int
    previously_contacted: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    poutcome: str


# ---------------------------------------------------------------------
# Generic helpers - shared by both datasets, driven entirely by
# feature_stats.json's shape rather than any dataset-specific logic.
# ---------------------------------------------------------------------

def load_model_bundle(model_key, registry):
    if model_key not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown model_key '{model_key}'")

    config = registry[model_key]
    checkpoint_dir = config["checkpoint_dir"]

    model_path = os.path.join(checkpoint_dir, "model.pkl")
    stats_path = os.path.join(checkpoint_dir, "feature_stats.json")

    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        raise HTTPException(
            status_code=500,
            detail=f"Model artifacts missing for '{model_key}' at {checkpoint_dir}. Train it first.",
        )

    model = joblib.load(model_path)
    with open(stats_path, "r") as f:
        feature_stats = json.load(f)

    return model, feature_stats, config


def build_feature_vector(input_data: BaseModel, feature_stats):
    numeric_features = feature_stats["numeric_features"]
    categorical_features = feature_stats["categorical_features"]
    model_feature_columns = feature_stats["model_feature_columns"]

    input_dict = input_data.dict()
    row = {}

    # Standardize numeric fields using the saved train-split mean/std
    for col, stats in numeric_features.items():
        if col not in input_dict:
            raise HTTPException(status_code=400, detail=f"Missing numeric field '{col}'")
        raw_value = input_dict[col]
        row[col] = (raw_value - stats["mean"]) / stats["std"] if stats["std"] > 0 else 0.0

    # One-hot encode categorical fields using the saved train-split categories
    for col, stats in categorical_features.items():
        if col not in input_dict:
            raise HTTPException(status_code=400, detail=f"Missing categorical field '{col}'")
        given_value = input_dict[col]
        known_categories = stats["categories"]

        if given_value not in known_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown category '{given_value}' for field '{col}'. "
                        f"Expected one of {known_categories}.",
            )

        for category in known_categories:
            dummy_col_name = f"{col}_{category}"
            if dummy_col_name in model_feature_columns:
                row[dummy_col_name] = 1 if given_value == category else 0

    # Assemble the final vector in the exact column order the model expects
    try:
        feature_vector = [row[col] for col in model_feature_columns]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Failed to build feature vector, missing column {e}")

    return np.array(feature_vector, dtype=float).reshape(1, -1)


def build_features_description(feature_stats):
    features_description = []
    for col, stats in feature_stats["numeric_features"].items():
        features_description.append({"name": col, "field_type": "numeric", "stats": stats})
    for col, stats in feature_stats["categorical_features"].items():
        features_description.append({
            "name": col,
            "field_type": "categorical",
            "categories": stats["categories"],
            "frequencies": stats["frequencies"],
        })
    return features_description


app = FastAPI(title="Demo Model Serving")


# ---------------------------------------------------------------------
# House Prices - regression
# ---------------------------------------------------------------------

@app.post("/house_prices/{model_key}/predict")
def predict_house_price(model_key: str, features: HousePriceFeatures):
    model, feature_stats, config = load_model_bundle(model_key, HOUSE_PRICE_MODELS)

    X = build_feature_vector(features, feature_stats)
    pred_standardized = model.predict(X)[0]

    target_stats = feature_stats["target"]
    pred_raw = pred_standardized * target_stats["std"] + target_stats["mean"]

    return {
        "model_key": model_key,
        "model_name": config["display_name"],
        "predicted_price": float(pred_raw),
    }


@app.get("/house_prices/{model_key}/describe")
def describe_house_price(model_key: str):
    _, feature_stats, config = load_model_bundle(model_key, HOUSE_PRICE_MODELS)

    example_request = {
        "area": 6000,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 1,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "prefarea": "no",
        "furnishingstatus": "semi-furnished",
    }

    return {
        "model_key": model_key,
        "model_name": config["display_name"],
        "job_type": config["job_type"],
        "modality": "tabular",
        "features": build_features_description(feature_stats),
        "how_to_call": {
            "predict_endpoint": f"POST /house_prices/{model_key}/predict",
            "example_request_body": example_request,
        },
        "output": {
            "field": "predicted_price",
            "description": "Predicted house price in original currency units (inverse-transformed from the model's standardized output).",
        },
    }


# ---------------------------------------------------------------------
# Bank Marketing - classification
# ---------------------------------------------------------------------

@app.post("/bank_marketing/{model_key}/predict")
def predict_bank_marketing(model_key: str, features: BankMarketingFeatures):
    model, feature_stats, config = load_model_bundle(model_key, BANK_MARKETING_MODELS)

    X = build_feature_vector(features, feature_stats)
    positive_probability = float(model.predict_proba(X)[0][1])

    target = feature_stats["target"]
    positive_class = target["positive_class"]
    negative_class = next(k for k in target["encoding"] if k != positive_class)
    predicted_probabilities = {positive_class: positive_probability, negative_class: 1.0 - positive_probability}
    predicted_class = positive_class if positive_probability >= 0.5 else negative_class

    return {
        "model_key": model_key,
        "model_name": config["display_name"],
        "predicted_class": predicted_class,
        "predicted_probabilities": predicted_probabilities,
        "confidence": max(predicted_probabilities.values()),
    }


@app.get("/bank_marketing/{model_key}/describe")
def describe_bank_marketing(model_key: str):
    _, feature_stats, config = load_model_bundle(model_key, BANK_MARKETING_MODELS)

    example_request = {
        "age": 42,
        "balance": 1500,
        "day": 15,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "previously_contacted": 0,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "poutcome": "unknown",
    }

    return {
        "model_key": model_key,
        "model_name": config["display_name"],
        "job_type": config["job_type"],
        "modality": "tabular",
        "features": build_features_description(feature_stats),
        "how_to_call": {
            "predict_endpoint": f"POST /bank_marketing/{model_key}/predict",
            "example_request_body": example_request,
        },
        "output": {
            "field": "predicted_probabilities",
            "reference_class": feature_stats["target"]["positive_class"],
            "description": (
                f"Predicted probability of each class. 'reference_class' "
                f"('{feature_stats['target']['positive_class']}' - the customer subscribes to a "
                f"term deposit) is the fixed target that explainability analysis measures "
                f"sensitivity/impact against."
            ),
        },
    }


# ---------------------------------------------------------------------
# BBC News - text classification
# ---------------------------------------------------------------------

class TextClassificationInput(BaseModel):
    text: str


def _get_by_index(mapping: dict, idx: int):
    """HF configs can round-trip id2label keys as either int or str
    depending on serialization path - handle both without guessing."""
    return mapping.get(idx, mapping.get(str(idx)))


def load_bbc_news_model(model_key: str) -> dict:
    if model_key not in BBC_NEWS_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model_key '{model_key}'")

    config = BBC_NEWS_MODELS[model_key]
    checkpoint_dir = config["checkpoint_dir"]

    if config["model_type"] == "tfidf":
        model_path = os.path.join(checkpoint_dir, "model.pkl")
        vectorizer_path = os.path.join(checkpoint_dir, "vectorizer.pkl")
        label_stats_path = os.path.join(checkpoint_dir, "label_stats.json")

        if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_stats_path)):
            raise HTTPException(
                status_code=500,
                detail=f"Model artifacts missing for '{model_key}' at {checkpoint_dir}. Train it first.",
            )

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        with open(label_stats_path, "r") as f:
            label_stats = json.load(f)

        return {"type": "tfidf", "model": model, "vectorizer": vectorizer, "label_stats": label_stats, "config": config}

    # model_type == "encoder"
    if not os.path.isdir(checkpoint_dir) or not os.listdir(checkpoint_dir):
        raise HTTPException(
            status_code=500,
            detail=f"Model artifacts missing for '{model_key}' at {checkpoint_dir}. Train it first.",
        )

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="torch/transformers are not installed on this server, required for the encoder model.",
        )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()

    return {"type": "encoder", "model": model, "tokenizer": tokenizer, "id2label": model.config.id2label, "config": config}


def predict_bbc_news_text(bundle: dict, text: str):
    if bundle["type"] == "tfidf":
        X = bundle["vectorizer"].transform([text])
        probs = bundle["model"].predict_proba(X)[0]
        classes = bundle["model"].classes_
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        predicted_class = str(classes[probs.argmax()])
        confidence = float(probs.max())
        return predicted_class, prob_dict, confidence

    # type == "encoder"
    import torch

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    id2label = bundle["id2label"]

    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).numpy()

    pred_idx = int(probs.argmax())
    predicted_class = _get_by_index(id2label, pred_idx)
    prob_dict = {_get_by_index(id2label, i): float(p) for i, p in enumerate(probs)}
    confidence = float(probs.max())

    return predicted_class, prob_dict, confidence


@app.post("/bbc_news/{model_key}/predict")
def predict_bbc_news(model_key: str, features: TextClassificationInput):
    bundle = load_bbc_news_model(model_key)
    predicted_class, prob_dict, confidence = predict_bbc_news_text(bundle, features.text)

    return {
        "model_key": model_key,
        "model_name": bundle["config"]["display_name"],
        "predicted_class": predicted_class,
        "predicted_probabilities": prob_dict,
        "confidence": confidence,
    }


@app.get("/bbc_news/{model_key}/describe")
def describe_bbc_news(model_key: str):
    bundle = load_bbc_news_model(model_key)
    config = bundle["config"]

    if bundle["type"] == "tfidf":
        categories = bundle["label_stats"]["categories"]
        category_frequencies = bundle["label_stats"]["frequencies"]
    else:
        id2label = bundle["id2label"]
        sorted_keys = sorted(id2label.keys(), key=lambda k: int(k))
        categories = [id2label[k] for k in sorted_keys]
        category_frequencies = None  # not tracked in the encoder checkpoint

    example_request = {
        "text": "The team secured a dramatic victory in the final minutes of the championship match."
    }

    return {
        "model_key": model_key,
        "model_name": config["display_name"],
        "job_type": config["job_type"],
        "modality": "text",
        "input_field": {
            "name": "text",
            "field_type": "text",
            "description": "Raw news article text to classify.",
        },
        "categories": categories,
        "category_frequencies": category_frequencies,
        "how_to_call": {
            "predict_endpoint": f"POST /bbc_news/{model_key}/predict",
            "example_request_body": example_request,
        },
        "output": {
            "field": "predicted_probabilities",
            "reference_class": config["reference_class"],
            "description": (
                f"Predicted probability of each category. 'reference_class' "
                f"('{config['reference_class']}') is the fixed target that explainability "
                f"analysis measures sensitivity/impact against; 'confidence' on /predict is "
                f"a separate, display-only value (the probability of whichever category was "
                f"actually predicted for that input)."
            ),
        },
    }