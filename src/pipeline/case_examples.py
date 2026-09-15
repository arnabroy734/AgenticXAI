"""
case_examples.py

Builds "sample case walkthrough" plots for a couple of individual
synthetic data points, reusing per-point results already computed
during the importance step - no extra explanation calls needed. Each
case captures the exact input values and the model's predicted output,
so the report can show them deterministically rather than relying on
the LLM to transcribe (and potentially misstate) the numbers.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


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