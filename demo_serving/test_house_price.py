"""
test.py

Hits the running House Price serving API with sample data points to
sanity-check both endpoints (predict, describe) for both models.

Make sure the server is already running (see run_server.sh) before
running this script.

Usage:
    python test.py
"""

import json
import requests

BASE_URL = "http://localhost:8000"

SAMPLE_POINTS = [
    {
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
    },
    {
        "area": 3500,
        "bedrooms": 2,
        "bathrooms": 1,
        "stories": 1,
        "parking": 0,
        "mainroad": "no",
        "guestroom": "no",
        "basement": "yes",
        "hotwaterheating": "no",
        "airconditioning": "no",
        "prefarea": "no",
        "furnishingstatus": "unfurnished",
    },
]

MODEL_KEYS = ["linear", "ann"]


def test_describe(model_key):
    print(f"\n--- GET /house_prices/{model_key}/describe ---")
    r = requests.get(f"{BASE_URL}/house_prices/{model_key}/describe")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print(json.dumps(r.json(), indent=2))
    else:
        print(r.json())


def test_predict(model_key, sample, index):
    print(f"\n--- POST /house_prices/{model_key}/predict (sample {index}) ---")
    r = requests.post(f"{BASE_URL}/house_prices/{model_key}/predict", json=sample)
    print(f"Status: {r.status_code}")
    print(r.json())


def test_invalid_category(model_key):
    print(f"\n--- POST /house_prices/{model_key}/predict (invalid category) ---")
    bad_sample = dict(SAMPLE_POINTS[0])
    bad_sample["furnishingstatus"] = "luxury"
    r = requests.post(f"{BASE_URL}/house_prices/{model_key}/predict", json=bad_sample)
    print(f"Status: {r.status_code}")
    print(r.json())


def main():
    for model_key in MODEL_KEYS:
        test_describe(model_key)
        for i, sample in enumerate(SAMPLE_POINTS, start=1):
            test_predict(model_key, sample, i)

    # Run the invalid-category check once, against the linear model
    test_invalid_category("linear")

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()