"""
report.py

Builds the report in two layers:
  - Deterministic chart generation (never delegated to the LLM, so the
    file path embedded in the report is always correct - see the
    marker-substitution approach below) and a deterministic header
    (model tested, endpoint, task type, data points analyzed,
    timestamp) - both generated directly by code so exact facts and
    numbers can never be misstated.
  - LLM-written narrative text around three markers, which are
    substituted with real charts (and, for the sample-case marker, real
    input/output details) after the LLM responds - the LLM never
    controls a file path or a data value itself.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..llm_client import call_llm_text
from ..report_header import build_header_block  # noqa: F401 - re-exported for backend.py

CHART_MARKER = "{{FEATURE_IMPORTANCE_CHART}}"
SAMPLE_CASES_MARKER = "{{SAMPLE_CASES}}"
IMPACT_MARKER = "{{FEATURE_IMPACT_CHARTS}}"


# ---------------------------------------------------------------------
# Deterministic chart generation
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# LLM report body (build_header_block is shared - see ../report_header.py)
# ---------------------------------------------------------------------

def build_report_json(model_name: str, job_type: str, top_numeric: list, top_categorical: list,
                       num_points: int, pdp_results: list = None, num_sample_cases: int = 0) -> dict:
    """
    Field names are deliberately generic (no 'lime', 'pdp', 'attribution',
    etc.) - this JSON is fed directly into the report LLM's prompt, so a
    method name that never appears here cannot leak into the output.
    """
    report = {
        "model_name": model_name,
        "job_type": job_type,
        "num_data_points_analyzed": num_points,
        "num_sample_cases_included": num_sample_cases,
        "top_numeric_features": [{"name": n, "sensitivity_score": round(v, 4)} for n, v in top_numeric],
        "top_categorical_features": [{"name": n, "sensitivity_score": round(v, 4)} for n, v in top_categorical],
    }

    if pdp_results:
        report["feature_impact_details"] = [
            {"feature_name": r["result"]["feature_name"], "feature_type": r["result"]["feature_type"]}
            for r in pdp_results
        ]

    return report


def generate_report_markdown(report_json: dict) -> str:
    system_prompt = (
        "You are writing a short, professional model explainability report for a business "
        "audience (e.g. a bank reviewing an internal model). Be factual and concise, and do "
        "not speculate beyond what the data shows. Do not write a title or a details/metadata "
        "section yourself - that has already been prepared separately.\n\n"
        "CRITICAL - methodology confidentiality: this analysis uses internal proprietary "
        "techniques. Do NOT mention, name, or describe the specific analytical methods used - "
        "never use the terms LIME, SHAP, PDP, partial dependence, permutation importance, "
        "surrogate model, perturbation, or any other specific algorithm name, anywhere in the "
        "report. Describe findings in plain business language instead.\n\n"
        "Section 1 - Sensitivity explanation: before showing the ranking chart, explain in "
        "plain language what a sensitivity score means - e.g. 'a sensitivity score shows how "
        "strongly each factor influences the model's output; a higher score means that factor "
        "has a bigger effect on the prediction.' The sensitivity_score values are already "
        "normalized to a 0-1 scale representing each feature's share of total measured "
        "sensitivity - you may describe these as relative shares or approximate percentages. "
        f"Then insert the exact marker '{CHART_MARKER}' alone on its own line.\n\n"
        "Section 2 - Sample cases: write one or two introductory sentences explaining that the "
        "following examples show specific real scenarios and what most influenced each "
        "prediction. Do NOT invent or state any specific input values or predicted numbers "
        f"yourself - they will be inserted automatically. Insert the exact marker "
        f"'{SAMPLE_CASES_MARKER}' alone on its own line immediately after your introductory text.\n\n"
        "Section 3 - Feature impact: explain in plain language what these charts show - e.g. "
        "'the charts below show how the predicted output changes as each factor varies, while "
        f"other factors remain typical of the dataset.' Then insert the exact marker "
        f"'{IMPACT_MARKER}' alone on its own line.\n\n"
        "Close with a brief note on how these results should be used (e.g. as a starting point "
        "for review, not a final verdict)."
    )

    user_prompt = f"Write the report body based on this data:\n\n{report_json}"

    return call_llm_text(system_prompt, user_prompt)


def _side_by_side_html(cells_html: list) -> str:
    """
    Arranges a list of pre-built inner HTML cell strings two-per-row.
    Uses an HTML table rather than CSS flexbox: wkhtmltopdf renders with
    an old, frozen WebKit engine whose flexbox support is broken/absent,
    so a flex layout that looks correct in a normal browser or the raw
    .md preview silently collapses to stacked blocks in the PDF output.
    Tables are reliably supported even by very old rendering engines.
    """
    rows = []
    for i in range(0, len(cells_html), 2):
        pair = cells_html[i:i + 2]
        tds = "".join(
            f'<td style="width:50%; padding:8px; vertical-align:top;">{cell}</td>' for cell in pair
        )
        if len(pair) == 1:
            tds += '<td style="width:50%;"></td>'
        row = f'<table style="width:100%; border-collapse:collapse;"><tr>{tds}</tr></table>'
        rows.append(row)
    return "\n\n".join(rows)


def _format_instance_html(instance: dict) -> str:
    def _fmt(value):
        if isinstance(value, float):
            return f"{value:,.2f}"
        return value

    items = "".join(f"<li><strong>{k}</strong>: {_fmt(v)}</li>" for k, v in instance.items())
    return f'<ul style="font-size:12px; margin:4px 0; padding-left:18px;">{items}</ul>'


def _build_sample_case_cells(sample_cases: list, job_type: str) -> list:
    output_label = "Predicted probability" if job_type == "classification" else "Predicted value"
    cells = []
    for i, case in enumerate(sample_cases, start=1):
        pred = case["prediction"]
        pred_str = f"{pred:.3f}" if job_type == "classification" else f"{pred:,.2f}"
        instance_html = _format_instance_html(case["instance"])
        cell = (
            '<div style="border:1px solid #e2e8f0; border-radius:6px; padding:8px;">'
            f'<img src="{case["plot_filename"]}" style="width:100%;">'
            f'<div style="font-size:13px; margin-top:8px;"><strong>Input values:</strong></div>'
            f'{instance_html}'
            f'<div style="font-size:13px; margin-top:6px;"><strong>{output_label}:</strong> {pred_str}</div>'
            '</div>'
        )
        cells.append(cell)
    return cells


def _build_impact_chart_cells(pdp_results: list) -> list:
    cells = []
    for r in pdp_results:
        feature_name = r["result"]["feature_name"]
        cell = (
            f'<img src="{r["plot_filename"]}" style="width:100%;">'
            f'<div style="font-size:13px; text-align:center; margin-top:4px;"><strong>{feature_name}</strong></div>'
        )
        cells.append(cell)
    return cells


def save_report(markdown_text: str, chart_path: str, pdp_results: list, sample_cases: list,
                 job_type: str, header_block: str, output_path: str) -> str:
    chart_relative_path = os.path.basename(chart_path)
    final_text = markdown_text.replace(CHART_MARKER, f"![Feature sensitivity]({chart_relative_path})")

    if SAMPLE_CASES_MARKER in final_text:
        cells = _build_sample_case_cells(sample_cases, job_type) if sample_cases else []
        final_text = final_text.replace(SAMPLE_CASES_MARKER, _side_by_side_html(cells))

    if IMPACT_MARKER in final_text:
        cells = _build_impact_chart_cells(pdp_results) if pdp_results else []
        final_text = final_text.replace(IMPACT_MARKER, _side_by_side_html(cells))

    final_text = header_block + final_text

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(final_text)

    return output_path
