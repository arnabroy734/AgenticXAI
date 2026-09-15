"""
lime.py

From-scratch implementation of LIME (Local Interpretable Model-agnostic
Explanations) for tabular data, operating entirely in raw feature space.

Algorithm (per-instance):
    1. Perturb the instance: numeric features get Gaussian noise scaled
       by their training std; categorical features are swapped to a
       different category (sampled by training frequency) with some
       probability.
    2. Convert each perturbed sample to an interpretable representation:
       numeric -> standardized value, categorical -> binary
       "same as original instance" indicator.
    3. Query predict_fn on every perturbed sample (raw space).
    4. Weight each sample by its proximity to the original instance,
       measured as Euclidean distance in the interpretable representation.
    5. Fit a weighted ridge regression on (interpretable representation -> prediction).
       The fitted coefficients are the local feature attributions.

predict_fn contract:
    For regression: takes a raw feature dict, returns the predicted value.
    For classification: takes a raw feature dict, returns the predicted
    probability of the class of interest (a continuous score, not a
    hard label) - this is what LIME is actually explaining.
"""

from typing import Callable, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .schema_utils import validate_feature_stats, get_numeric_perturbation_std


def _sample_one_perturbation(x, numeric_features, categorical_features, kappa, categorical_flip_prob, rng):
    z = dict(x)

    for name, stats in numeric_features.items():
        std = get_numeric_perturbation_std(stats, scale=kappa)
        z[name] = x[name] + rng.normal(0, std)

    for name, stats in categorical_features.items():
        if rng.random() < categorical_flip_prob:
            categories = stats["categories"]
            frequencies = stats.get("frequencies", {})
            probs = np.array([frequencies.get(c, 0.0) for c in categories], dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else None
            z[name] = rng.choice(categories, p=probs)
        # else: leave z[name] as the original x[name] (already copied)

    return z


def _interpretable_repr(z, x, numeric_features, categorical_features):
    vec = []
    for name, stats in numeric_features.items():
        mean, std = stats["mean"], stats["std"]
        vec.append((z[name] - mean) / std)
    for name in categorical_features:
        vec.append(1.0 if z[name] == x[name] else 0.0)
    return np.array(vec, dtype=float)


def explain_instance(
    x: dict,
    predict_fn: Callable[[dict], float],
    feature_stats: dict,
    num_samples: int = 500,
    kappa: float = 1.0,
    categorical_flip_prob: float = 0.5,
    kernel_width: Optional[float] = None,
    ridge_alpha: float = 0.01,
    random_state: Optional[int] = None,
) -> dict:
    """
    Explains a single prediction for instance x.

    Returns a dict with the instance, the model's actual prediction,
    per-feature local attributions (ridge coefficients), the local
    surrogate's intercept, and its weighted R^2 (a confidence signal -
    low R^2 means the linear surrogate did not fit the local
    neighborhood well, and the attributions should not be trusted at
    face value).
    """
    numeric_features = feature_stats.get("numeric_features", {})
    categorical_features = feature_stats.get("categorical_features", {})
    feature_names = list(numeric_features.keys()) + list(categorical_features.keys())

    validate_feature_stats(feature_stats, feature_names)

    missing_in_x = [f for f in feature_names if f not in x]
    if missing_in_x:
        raise ValueError(f"Instance x is missing required field(s): {missing_in_x}")

    rng = np.random.default_rng(random_state)

    if kernel_width is None:
        kernel_width = 0.75 * np.sqrt(len(feature_names))

    perturbed_raw = [dict(x)]  # include the original instance itself as one sample
    for _ in range(num_samples):
        perturbed_raw.append(
            _sample_one_perturbation(x, numeric_features, categorical_features, kappa, categorical_flip_prob, rng)
        )

    x_repr = _interpretable_repr(x, x, numeric_features, categorical_features)

    Z_repr = []
    y = []
    for z in perturbed_raw:
        Z_repr.append(_interpretable_repr(z, x, numeric_features, categorical_features))
        y.append(predict_fn(z))

    Z_repr = np.vstack(Z_repr)
    y = np.array(y, dtype=float)

    distances = np.linalg.norm(Z_repr - x_repr, axis=1)
    weights = np.exp(-(distances**2) / (kernel_width**2))

    surrogate = Ridge(alpha=ridge_alpha)
    surrogate.fit(Z_repr, y, sample_weight=weights)

    y_pred = surrogate.predict(Z_repr)
    local_r2 = r2_score(y, y_pred, sample_weight=weights)

    attributions = {name: float(coef) for name, coef in zip(feature_names, surrogate.coef_)}

    return {
        "instance": x,
        "prediction": float(predict_fn(x)),
        "attributions": attributions,
        "local_model_intercept": float(surrogate.intercept_),
        "local_model_r2": float(local_r2),
        "num_samples": num_samples,
    }