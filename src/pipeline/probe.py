"""
probe.py

Step 1: probe a model's /describe endpoint to learn its job type,
feature schema, and which field name to read from /predict responses.

Generic by design - works for any model serving both /describe and
/predict under the same URL prefix (the convention already used by
demo_serving/app.py), not hardcoded to house prices specifically.
"""

import requests


def probe_model(describe_url: str) -> dict:
    """
    Returns:
        {
            "job_type": "regression" or "classification",
            "feature_stats": {"numeric_features": {...}, "categorical_features": {...}},
            "output_field": "predicted_price",   # field name to read from /predict responses
            "predict_url": "<derived by replacing the /describe suffix with /predict>",
        }
    """
    r = requests.get(describe_url)
    r.raise_for_status()
    body = r.json()

    numeric_features, categorical_features = {}, {}
    for f in body["features"]:
        if f["field_type"] == "numeric":
            numeric_features[f["name"]] = f["stats"]
        else:
            categorical_features[f["name"]] = {
                "categories": f["categories"],
                "frequencies": f["frequencies"],
            }

    feature_stats = {"numeric_features": numeric_features, "categorical_features": categorical_features}

    if not describe_url.endswith("/describe"):
        raise ValueError(f"Expected describe_url to end with '/describe', got: {describe_url}")
    predict_url = describe_url[: -len("/describe")] + "/predict"

    return {
        "job_type": body["job_type"],
        "model_name": body["model_name"], 
        "feature_stats": feature_stats,
        "output_field": body["output"]["field"],
        "predict_url": predict_url,
    }