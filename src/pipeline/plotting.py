"""
plotting.py

Deterministic plot generation for the report - never delegated to the
LLM, so the file path embedded in the report is always correct (see
report_generator.py's marker-substitution approach).
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def save_importance_plot(top_numeric: list, top_categorical: list, output_path: str,
                          title: str = "Feature sensitivity ranking"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    combined = [(name, value, "numeric") for name, value in top_numeric] + \
               [(name, value, "categorical") for name, value in top_categorical]
    combined.sort(key=lambda x: x[1])

    names = [c[0] for c in combined]
    values = [c[1] for c in combined]
    colors = ["#185FA5" if c[2] == "numeric" else "#7F77DD" for c in combined]

    plt.figure(figsize=(7, 5))
    plt.barh(names, values, color=colors)
    plt.xlabel("Normalized sensitivity score (0-1)")
    plt.title(title)

    legend_elements = [
        Patch(facecolor="#185FA5", label="Numeric"),
        Patch(facecolor="#7F77DD", label="Categorical"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_pdp_plot(pdp_result: dict, output_path: str, title: str = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    grid_values = pdp_result["grid_values"]
    avg_preds = pdp_result["average_predictions"]

    plt.figure(figsize=(6, 4))
    if pdp_result["feature_type"] == "numeric":
        plt.plot(grid_values, avg_preds, marker="o", color="#185FA5")
    else:
        plt.bar([str(v) for v in grid_values], avg_preds, color="#7F77DD")

    plt.xlabel(pdp_result["feature_name"])
    plt.ylabel("Average predicted output")
    plt.title(title or f"Impact of {pdp_result['feature_name']} on prediction")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path