"""
backend.py

Text-classification modality backend. Implements the same contract as
tabular/backend.py:

    run(base_url, model_key, describe_body, num_points,
        num_lime_samples, working_dir) -> {
            "model_name": str,
            "job_type": "classification",
            "reference_class": None,
            "num_points_analyzed": int,
            "artifact_filenames": {generic_key: filename_within_working_dir},
        }

num_points here means "total number of synthetic documents to
generate", stratified as evenly as possible across every category the
model declares (see model_client.generate_synthetic_documents - e.g.
100 documents over 4 categories -> 25 per category). num_lime_samples
has no text analogue (hierarchical occlusion is exhaustive, not
sampled) and is accepted but ignored, purely for signature
compatibility with the shared dispatcher in run_pipeline.py, which
calls every backend with the same keyword arguments.

reference_class is always None in this backend's result - unlike
tabular's single fixed positive_class, every document here is
explained against its OWN model-predicted class (see explain.py), so
there is no one class name to report at the run level.

Orchestrates:
    1. Parse the already-fetched /describe response
    2. Generate synthetic news-article documents (LLM), stratified by category
    3. For every document: get its own predicted class, run hierarchical
       sentence+n-gram LIME fixed to that class
    4. Render every document's highlighting, grouped by intended category
    5. Generate the report (LLM intro + deterministic task description
       and documents) and convert to PDF - no chart/bar-graph step
"""

import json
import logging
import os

from .model_client import parse_describe_body, generate_synthetic_documents
from .explain import explain_documents
from .report import (
    build_task_description_block,
    build_classified_documents_html,
    build_report_json,
    generate_report_markdown,
    save_report,
)
from ..report_header import build_header_block
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
    categories = probe_result["categories"]
    logger.info("categories=%s", categories)

    logger.info(
        "Generating %d synthetic documents, stratified across %d categories",
        num_points, len(categories),
    )
    synthetic_docs = generate_synthetic_documents(categories, num_docs=num_points, max_retries=3)

    logger.info(
        "Running hierarchical text LIME across %d document(s), each against its own predicted class",
        len(synthetic_docs),
    )
    per_doc_results = explain_documents(
        synthetic_docs, probe_result["predict_url"], top_k_sentences=1, include_bigrams=True
    )

    artifact_filenames = {}

    report_json = build_report_json(model_name, categories, per_doc_results)
    report_json_filename = f"{model_key}_report.json"
    report_json_path = os.path.join(working_dir, report_json_filename)
    with open(report_json_path, "w") as f:
        json.dump(report_json, f, indent=2)
    artifact_filenames["report_json"] = report_json_filename

    logger.info("Generating report via LLM")
    markdown_text = generate_report_markdown(report_json)
    header_block = build_header_block(base_url, model_name, "classification", len(synthetic_docs))
    task_description_block = build_task_description_block(categories)
    classified_documents_html = build_classified_documents_html(per_doc_results, categories)

    report_md_filename = f"{model_key}_report.md"
    report_md_path = os.path.join(working_dir, report_md_filename)
    save_report(markdown_text, classified_documents_html, task_description_block, header_block, report_md_path)
    artifact_filenames["report_markdown"] = report_md_filename

    logger.info("Converting report to PDF")
    report_pdf_filename = f"{model_key}_report.pdf"
    report_pdf_path = os.path.join(working_dir, report_pdf_filename)
    convert_markdown_to_pdf(report_md_path, report_pdf_path)
    artifact_filenames["report_pdf"] = report_pdf_filename

    return {
        "model_name": model_name,
        "job_type": "classification",
        "reference_class": None,
        "num_points_analyzed": len(synthetic_docs),
        "artifact_filenames": artifact_filenames,
    }
