"""
app.py

Serves the House Price models (ElasticNet linear regression and the
2-layer ANN) behind a common FastAPI app. Each model gets two endpoints:

    POST /house_prices/{model_key}/predict   - run a prediction
    GET  /house_prices/{model_key}/describe   - schema, feature stats, job type

where model_key is "linear" or "ann".

Usage:
    uvicorn app:app --reload
"""

import os
import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "demo_models", "checkpoints")

MODEL_REGISTRY = {
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


def load_model_bundle(model_key):
    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model_key '{model_key}'")

    config = MODEL_REGISTRY[model_key]
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


def build_feature_vector(input_data: HousePriceFeatures, feature_stats):
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
        row[col] = (raw_value - stats["mean"]) / stats["std"]

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


app = FastAPI(title="House Price Model Serving")


@app.post("/house_prices/{model_key}/predict")
def predict(model_key: str, features: HousePriceFeatures):
    model, feature_stats, config = load_model_bundle(model_key)

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
def describe(model_key: str):
    _, feature_stats, config = load_model_bundle(model_key)

    numeric_features = feature_stats["numeric_features"]
    categorical_features = feature_stats["categorical_features"]

    features_description = []
    for col, stats in numeric_features.items():
        features_description.append({
            "name": col,
            "field_type": "numeric",
            "stats": stats,
        })
    for col, stats in categorical_features.items():
        features_description.append({
            "name": col,
            "field_type": "categorical",
            "categories": stats["categories"],
            "frequencies": stats["frequencies"],
        })

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
        "features": features_description,
        "how_to_call": {
            "predict_endpoint": f"POST /house_prices/{model_key}/predict",
            "example_request_body": example_request,
        },
        "output": {
            "field": "predicted_price",
            "description": "Predicted house price in original currency units (inverse-transformed from the model's standardized output).",
        },
    }