"""
importance.py

Step 5: runs LIME across every synthetic data point, averages the
absolute attribution per feature across all points, and selects the
top 5 numeric + top 5 categorical features by that average.
"""

import numpy as np

from xai import lime as lime_module


def compute_average_importance(
    synthetic_rows: list,
    predict_fn,
    feature_stats: dict,
    num_lime_samples: int = 500,
    top_n: int = 5,
) -> dict:
    feature_names = list(feature_stats["numeric_features"].keys()) + list(feature_stats["categorical_features"].keys())
    abs_attributions = {name: [] for name in feature_names}

    per_point_results = []
    for i, row in enumerate(synthetic_rows):
        result = lime_module.explain_instance(
            x=row,
            predict_fn=predict_fn,
            feature_stats=feature_stats,
            num_samples=num_lime_samples,
            random_state=i,  # vary seed per point for genuine diversity across the batch
        )
        per_point_results.append(result)
        for name, value in result["attributions"].items():
            abs_attributions[name].append(abs(value))

    average_importance = {name: float(np.mean(values)) for name, values in abs_attributions.items()}

    numeric_names = set(feature_stats["numeric_features"].keys())
    categorical_names = set(feature_stats["categorical_features"].keys())

    numeric_ranked = sorted(
        [(k, v) for k, v in average_importance.items() if k in numeric_names],
        key=lambda kv: kv[1],
        reverse=True,
    )
    categorical_ranked = sorted(
        [(k, v) for k, v in average_importance.items() if k in categorical_names],
        key=lambda kv: kv[1],
        reverse=True,
    )

    return {
        "average_importance": average_importance,
        "top_numeric": numeric_ranked[:top_n],
        "top_categorical": categorical_ranked[:top_n],
        "num_points_explained": len(synthetic_rows),
        "per_point_results": per_point_results,
    }


def normalize_importance_scores(top_numeric: list, top_categorical: list) -> tuple:
    """
    Rescales raw attribution-magnitude scores to a 0-1 proportional
    share of the total across both lists combined (i.e. each score
    becomes "this feature accounts for X% of the combined sensitivity
    shown here"). Ordering is unaffected, since this is a positive
    linear rescaling - only the selection of which features to run
    downstream analysis on should happen BEFORE normalization is
    irrelevant either way, since order is preserved.
    """
    total = sum(v for _, v in top_numeric) + sum(v for _, v in top_categorical)
    if total <= 0:
        return top_numeric, top_categorical

    normalized_numeric = [(name, value / total) for name, value in top_numeric]
    normalized_categorical = [(name, value / total) for name, value in top_categorical]
    return normalized_numeric, normalized_categorical