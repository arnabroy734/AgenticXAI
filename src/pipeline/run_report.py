"""
run_report.py

Orchestrates the full deterministic pipeline:
    1. Probe /describe
    2. Generate + validate synthetic data (LLM, with retries)
    3. (validation happens inside step 2)
    4. Build predict_fn (works for regression and classification alike)
    5. Run LIME across synthetic points, rank top 5 numeric + top 5 categorical
    6. Save the consolidated JSON + a deterministic importance plot
    7. Generate the report (LLM) and embed the plot

Usage:
    python run_report.py <base_url> <model_key> [num_points] [num_lime_samples]

Example:
    python run_report.py http://0.0.0.0:8000/house_prices linear
"""

import os
import re
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.probe import probe_model
from pipeline.predict_client import make_predict_fn
from pipeline.synthetic_data import generate_synthetic_data
from pipeline.importance import compute_average_importance, normalize_importance_scores
from pipeline.pdp_step import run_pdp_on_top_features
from pipeline.case_examples import build_sample_cases
from pipeline.plotting import save_importance_plot
from pipeline.report_generator import build_header_block, build_report_json, generate_report_markdown, save_report
from pipeline.pdf_export import convert_markdown_to_pdf


def _sanitize_for_path(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def _build_output_dir(base_url: str, model_key: str) -> str:
    subfolder = f"{_sanitize_for_path(base_url)}__{_sanitize_for_path(model_key)}"
    output_dir = os.path.join(os.getcwd(), "explainability_reports", subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run(base_url: str, model_key: str, num_points: int = 100, num_lime_samples: int = 500) -> str:
    describe_url = f"{base_url}/{model_key}/describe"
    output_dir = _build_output_dir(base_url, model_key)
    print(f"Output directory: {output_dir}")

    print(f"\nStep 1: probing {describe_url} ...")
    probe_result = probe_model(describe_url)
    model_name = probe_result["model_name"]
    feature_stats = probe_result["feature_stats"]
    job_type = probe_result["job_type"]
    print(f"  job_type = {job_type}")

    print(f"\nStep 2-3: generating + validating {num_points} synthetic data points ...")
    synthetic_rows = generate_synthetic_data(feature_stats, num_points=num_points, max_retries=3)

    print("\nStep 4: building predict_fn ...")
    predict_fn = make_predict_fn(probe_result["predict_url"], probe_result["output_field"], feature_stats)

    print(f"\nStep 5: running LIME across {len(synthetic_rows)} synthetic points ...")
    importance_result = compute_average_importance(
        synthetic_rows, predict_fn, feature_stats, num_lime_samples=num_lime_samples, top_n=5
    )

    top_numeric, top_categorical = normalize_importance_scores(
        importance_result["top_numeric"], importance_result["top_categorical"]
    )
    print(f"  Top numeric (normalized): {top_numeric}")
    print(f"  Top categorical (normalized): {top_categorical}")

    print("\nStep 6: saving consolidated JSON + sensitivity plot ...")
    chart_path = os.path.join(output_dir, f"{model_key}_sensitivity_chart.png")
    save_importance_plot(top_numeric, top_categorical, chart_path)
    print(f"  Saved plot -> {chart_path}")

    print("\nStep 6b: running impact analysis on top 4 numeric + top 4 categorical features ...")
    pdp_results = run_pdp_on_top_features(
        top_numeric=top_numeric,
        top_categorical=top_categorical,
        predict_fn=predict_fn,
        feature_stats=feature_stats,
        background_data=synthetic_rows,
        output_dir=output_dir,
        model_key=model_key,
        top_n=4,
    )
    print(f"  Generated {len(pdp_results)} plot(s): {[r['result']['feature_name'] for r in pdp_results]}")

    print("\nStep 6c: building sample case walkthroughs ...")
    sample_cases = build_sample_cases(
        importance_result["per_point_results"], output_dir=output_dir, model_key=model_key, num_cases=2
    )
    print(f"  Built {len(sample_cases)} sample case(s)")

    report_json = build_report_json(
        model_name, job_type, top_numeric, top_categorical, len(synthetic_rows), pdp_results, len(sample_cases),
    )
    report_json_path = os.path.join(output_dir, f"{model_key}_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"  Saved report JSON -> {report_json_path}")

    print("\nStep 7: generating report via LLM ...")
    markdown_text = generate_report_markdown(report_json)
    header_block = build_header_block(base_url, model_name, job_type, len(synthetic_rows))

    report_md_path = os.path.join(output_dir, f"{model_key}_report.md")
    save_report(markdown_text, chart_path, pdp_results, sample_cases, job_type, header_block, report_md_path)
    print(f"  Saved report -> {report_md_path}")

    print("\nStep 8: converting report to PDF ...")
    report_pdf_path = os.path.join(output_dir, f"{model_key}_report.pdf")
    convert_markdown_to_pdf(report_md_path, report_pdf_path)
    print(f"  Saved PDF -> {report_pdf_path}")

    return report_pdf_path


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model_key = sys.argv[2] if len(sys.argv) > 2 else "linear"
    num_points = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    num_lime_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    run(base_url, model_key, num_points, num_lime_samples)