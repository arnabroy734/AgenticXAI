"""
backend.py

Tabular modality backend. Implements the contract run_pipeline.py
expects from any modality backend:

    run(base_url, model_key, describe_body, num_points,
        num_lime_samples, working_dir) -> {
            "model_name": str,
            "job_type": str,
            "reference_class": str or None,
            "num_points_analyzed": int,
            "artifact_filenames": {generic_key: filename_within_working_dir},
        }

Orchestrates the full deterministic pipeline for a tabular model:
    1. Parse the already-fetched /describe response
    2. Generate + validate synthetic data (LLM, with retries)
    3. (validation happens inside step 2)
    4. Build predict_fn (works for regression and classification alike)
    5. Run LIME across synthetic points, rank top 5 numeric + top 5 categorical
    6. Save the consolidated JSON + a deterministic importance plot
    7. Generate the report (LLM) and embed the plot
    8. Convert to PDF

A future non-tabular backend implements the same run(...) signature
and gets registered in run_pipeline.py's BACKENDS dict - nothing here
needs to change for that.
"""

import json
import logging
import os

from .model_client import parse_describe_body, make_predict_fn, generate_synthetic_data
from .explain import (
    compute_average_importance,
    normalize_importance_scores,
    run_pdp_on_top_features,
    build_sample_cases,
)
from .report import (
    save_importance_plot,
    build_header_block,
    build_report_json,
    generate_report_markdown,
    save_report,
)
from ..pdf_export import convert_markdown_to_pdf

logger = logging.getLogger(__name__)


def run(
    base_url: str,
    model_key: str,
    describe_body: dict,
    num_points: int,
    num_lime_samples: int,
    working_dir: str,
) -> dict:
    probe_result = parse_describe_body(describe_body, base_url, model_key)
    model_name = probe_result["model_name"]
    feature_stats = probe_result["feature_stats"]
    job_type = probe_result["job_type"]
    reference_class = probe_result["reference_class"]
    logger.info(
        "job_type=%s%s", job_type, f" reference_class={reference_class}" if reference_class else ""
    )

    logger.info("Generating + validating %d synthetic data points", num_points)
    synthetic_rows = generate_synthetic_data(feature_stats, num_points=num_points, max_retries=3)

    logger.info("Building predict_fn")
    predict_fn = make_predict_fn(
        probe_result["predict_url"],
        probe_result["output_field"],
        feature_stats,
        job_type=job_type,
        reference_class=reference_class,
    )

    logger.info("Running LIME across %d synthetic points", len(synthetic_rows))
    importance_result = compute_average_importance(
        synthetic_rows, predict_fn, feature_stats, num_lime_samples=num_lime_samples, top_n=5
    )
    top_numeric, top_categorical = normalize_importance_scores(
        importance_result["top_numeric"], importance_result["top_categorical"]
    )
    logger.info("Top numeric (normalized): %s", top_numeric)
    logger.info("Top categorical (normalized): %s", top_categorical)

    artifact_filenames = {}

    chart_filename = f"{model_key}_sensitivity_chart.png"
    chart_path = os.path.join(working_dir, chart_filename)
    save_importance_plot(top_numeric, top_categorical, chart_path)
    artifact_filenames["sensitivity_chart"] = chart_filename

    logger.info("Running impact analysis on top 4 numeric + top 4 categorical features")
    pdp_results = run_pdp_on_top_features(
        top_numeric=top_numeric,
        top_categorical=top_categorical,
        predict_fn=predict_fn,
        feature_stats=feature_stats,
        background_data=synthetic_rows,
        output_dir=working_dir,
        model_key=model_key,
        top_n=4,
    )
    for r in pdp_results:
        artifact_filenames[f"impact_chart_{r['result']['feature_name']}"] = r["plot_filename"]

    logger.info("Building sample case walkthroughs")
    sample_cases = build_sample_cases(
        importance_result["per_point_results"], output_dir=working_dir, model_key=model_key, num_cases=2
    )
    for i, case in enumerate(sample_cases, start=1):
        artifact_filenames[f"sample_case_{i}"] = case["plot_filename"]

    report_json = build_report_json(
        model_name, job_type, top_numeric, top_categorical, len(synthetic_rows), pdp_results, len(sample_cases),
    )
    report_json_filename = f"{model_key}_report.json"
    report_json_path = os.path.join(working_dir, report_json_filename)
    with open(report_json_path, "w") as f:
        json.dump(report_json, f, indent=2)
    artifact_filenames["report_json"] = report_json_filename

    logger.info("Generating report via LLM")
    markdown_text = generate_report_markdown(report_json)
    header_block = build_header_block(
        base_url, model_name, job_type, len(synthetic_rows), reference_class=reference_class
    )

    report_md_filename = f"{model_key}_report.md"
    report_md_path = os.path.join(working_dir, report_md_filename)
    save_report(markdown_text, chart_path, pdp_results, sample_cases, job_type, header_block, report_md_path)
    artifact_filenames["report_markdown"] = report_md_filename

    logger.info("Converting report to PDF")
    report_pdf_filename = f"{model_key}_report.pdf"
    report_pdf_path = os.path.join(working_dir, report_pdf_filename)
    convert_markdown_to_pdf(report_md_path, report_pdf_path)
    artifact_filenames["report_pdf"] = report_pdf_filename

    return {
        "model_name": model_name,
        "job_type": job_type,
        "reference_class": reference_class,
        "num_points_analyzed": len(synthetic_rows),
        "artifact_filenames": artifact_filenames,
    }
