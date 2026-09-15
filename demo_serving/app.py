"""
app.py

Serves both the House Price models (regression) and the Bank Marketing
models (classification) behind one FastAPI app. Each model gets two
endpoints, under its own dataset prefix:

    POST /house_prices/{model_key}/predict     model_key: "linear" or "ann"
    GET  /house_prices/{model_key}/describe

    POST /bank_marketing/{model_key}/predict   model_key: "logistic", "rf", or "ann"
    GET  /bank_marketing/{model_key}/describe

The two datasets have different request schemas (different real-world
fields), so they get their own Pydantic models and route functions -
but both share the same generic model-loading and feature-vector-
building logic underneath, since that logic only depends on
feature_stats.json's shape, not on which dataset it came from.

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
    predicted_probability = float(model.predict_proba(X)[0][1])

    target = feature_stats["target"]
    positive_class = target["positive_class"]
    negative_class = next(k for k in target["encoding"] if k != positive_class)
    predicted_class = positive_class if predicted_probability >= 0.5 else negative_class

    return {
        "model_key": model_key,
        "model_name": config["display_name"],
        "predicted_class": predicted_class,
        "predicted_probability": predicted_probability,
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
        "features": build_features_description(feature_stats),
        "how_to_call": {
            "predict_endpoint": f"POST /bank_marketing/{model_key}/predict",
            "example_request_body": example_request,
        },
        "output": {
            "field": "predicted_probability",
            "description": (
                f"Predicted probability of the positive class "
                f"('{feature_stats['target']['positive_class']}' - the customer subscribes "
                f"to a term deposit)."
            ),
        },
    }