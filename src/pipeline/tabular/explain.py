"""
explain.py

Runs the actual explainability computation over tabular data:
  - Step 5: LIME across every synthetic data point, averaged into a
    top-N feature importance ranking.
  - Step 6: PDP on the top 4 numeric + top 4 categorical features
    from that ranking (importance ranks top 5 of each; PDP
    intentionally looks at a tighter top 4 for visual brevity), reusing
    the synthetic batch itself as PDP's background dataset.
  - Step 6c: 2 individual sample-case walkthroughs, reusing per-point
    LIME results already computed in step 5 - no extra explanation
    calls needed.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from xai import lime as lime_module
from xai import pdp as pdp_module
from .report import save_pdp_plot


# ---------------------------------------------------------------------
# Step 5: LIME importance ranking
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Step 6: PDP on top features
# ---------------------------------------------------------------------

def run_pdp_on_top_features(
    top_numeric: list,
    top_categorical: list,
    predict_fn,
    feature_stats: dict,
    background_data: list,
    output_dir: str,
    model_key: str,
    top_n: int = 4,
    grid_size: int = 20,
) -> list:
    selected_numeric = [name for name, _ in top_numeric[:top_n]]
    selected_categorical = [name for name, _ in top_categorical[:top_n]]
    selected_features = selected_numeric + selected_categorical

    results = []
    for feature_name in selected_features:
        pdp_result = pdp_module.compute_pdp(
            feature_name=feature_name,
            predict_fn=predict_fn,
            feature_stats=feature_stats,
            background_data=background_data,
            grid_size=grid_size,
        )

        plot_filename = f"{model_key}_feature_impact_{feature_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        save_pdp_plot(pdp_result, plot_path, title=f"Impact of {feature_name} on prediction")

        results.append({"result": pdp_result, "plot_filename": plot_filename})

    return results


# ---------------------------------------------------------------------
# Step 6c: sample-case walkthroughs
# ---------------------------------------------------------------------

def _normalize_to_unit_range(values: list) -> list:
    """Max-abs normalization: divides every value by the largest absolute
    value in the list, so the most influential feature lands at exactly
    +/-1 and everything else falls proportionally within [-1, 1], with
    sign (positive/negative direction) preserved."""
    max_abs = max(abs(v) for v in values) if values else 0
    if max_abs == 0:
        return [0.0 for _ in values]
    return [v / max_abs for v in values]


def _save_case_plot(result: dict, output_path: str, title: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    attributions = result["attributions"]
    names = list(attributions.keys())
    raw_values = list(attributions.values())
    values = _normalize_to_unit_range(raw_values)

    order = np.argsort(np.abs(values))
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in values]

    plt.figure(figsize=(6, 4.5))
    plt.barh(names, values, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlim(-1.1, 1.1)
    plt.title(title)
    plt.xlabel("Normalized influence on this prediction (-1 to 1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_sample_cases(per_point_results: list, output_dir: str, model_key: str, num_cases: int = 2) -> list:
    """
    Returns a list of dicts, one per selected case:
        {
            "plot_filename": "...",
            "instance": {...raw feature values...},
            "prediction": float,
        }
    Cases are spread evenly across the analyzed batch (not just the
    first N), for a more representative pair of examples.
    """
    if not per_point_results:
        return []

    count = min(num_cases, len(per_point_results))
    indices = sorted(set(np.linspace(0, len(per_point_results) - 1, num=count, dtype=int).tolist()))

    cases = []
    for case_num, idx in enumerate(indices, start=1):
        result = per_point_results[idx]
        plot_filename = f"{model_key}_sample_case_{case_num}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        _save_case_plot(result, plot_path, title=f"Sample case {case_num}")

        cases.append({
            "plot_filename": plot_filename,
            "instance": result["instance"],
            "prediction": result["prediction"],
        })

    return cases
