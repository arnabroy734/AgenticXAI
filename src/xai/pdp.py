"""
pdp.py

From-scratch implementation of Partial Dependence (PDP) for a single
feature at a time, operating entirely in raw feature space.

Algorithm:
    1. Build a grid of candidate values for the target feature:
       - Numeric: evenly spaced points across get_numeric_range(...)
         (p25/p75 -> min/max -> mean +/- 2*std fallback), default 20 points.
       - Categorical: every known category, in one pass.
    2. For each candidate value v:
        - Take the background batch (M real, raw rows).
        - Overwrite the target feature to v in every row (all other
          features in each row stay exactly as they were).
        - Call predict_fn on all M modified rows.
        - Average the M outputs -> one point on the PDP curve/bar chart.
    3. Return the grid of candidate values alongside their averaged
       predictions.

Unlike LIME, this is computed for ONE feature at a time - call it once
per feature you want a PDP for.

background_data must be real, raw rows (e.g. sampled from training
data) - not independently-sampled marginals per feature. Independent
marginal sampling can produce unrealistic combinations (e.g. a
1-bedroom house with 10,000 sq ft) since it ignores real correlations
between features.
"""

from typing import Callable

import numpy as np

from .schema_utils import validate_feature_stats, get_numeric_range


def compute_pdp(
    feature_name: str,
    predict_fn: Callable[[dict], float],
    feature_stats: dict,
    background_data: list,
    grid_size: int = 20,
) -> dict:
    """
    Computes the partial dependence of predict_fn's output on a single
    feature, averaged over background_data.

    feature_name: the single feature to compute PDP for.
    predict_fn: raw feature dict -> prediction (same contract as LIME).
    feature_stats: same shape as checkpoints/<model>/feature_stats.json.
    background_data: list of raw feature dicts - real rows representing
        realistic combinations of all other features.
    grid_size: number of points in the numeric grid (ignored for
        categorical features, which always use every known category).
    """
    numeric_features = feature_stats.get("numeric_features", {})
    categorical_features = feature_stats.get("categorical_features", {})

    validate_feature_stats(feature_stats, [feature_name])

    if not background_data:
        raise ValueError("background_data must be a non-empty list of raw feature dicts.")

    if feature_name in numeric_features:
        feature_type = "numeric"
        stats = numeric_features[feature_name]
        low, high = get_numeric_range(stats)
        grid_values = np.linspace(low, high, grid_size).tolist()
    else:
        feature_type = "categorical"
        grid_values = list(categorical_features[feature_name]["categories"])

    average_predictions = []
    for value in grid_values:
        predictions = []
        for row in background_data:
            modified_row = dict(row)
            modified_row[feature_name] = value
            predictions.append(predict_fn(modified_row))
        average_predictions.append(float(np.mean(predictions)))

    return {
        "feature_name": feature_name,
        "feature_type": feature_type,
        "grid_values": grid_values,
        "average_predictions": average_predictions,
        "background_size": len(background_data),
    }