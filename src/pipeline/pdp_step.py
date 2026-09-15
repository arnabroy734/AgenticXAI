"""
pdp_step.py

Runs PDP on a subset of features - by default, the top 4 numeric + top
4 categorical features identified by the importance step (importance.py
computes top 5 of each for the ranking report; PDP intentionally looks
at a tighter top 4 for visual brevity). Uses the synthetic data batch
itself as PDP's background dataset (it's already schema/bounds-validated
real rows, no separate generation needed). Saves one plot per feature.
"""

import os

from xai import pdp as pdp_module
from .plotting import save_pdp_plot


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