"""
report.py

Builds the text-modality report: a deterministic header (shared - see
../report_header.py), a deterministic task-description block, and
per-document two-tier highlighting grouped by category (never
delegated to the LLM, so the category names and highlighting are
always exactly what the model actually did) - with a short LLM-written
introduction around one marker, substituted with the highlighted
documents afterward, so the LLM never controls a file path, a category
name, or a data value.

No chart/bar-graph step and no PDP-equivalent "impact" section - the
report's entire content is the highlighted documents themselves.
"""

import os

from xai import lime_text as lime_text_module
from ..llm_client import call_llm_text

DOCUMENTS_MARKER = "{{CLASSIFIED_DOCUMENTS}}"


# ---------------------------------------------------------------------
# Deterministic task description + document rendering
# ---------------------------------------------------------------------

def build_task_description_block(categories: list) -> str:
    """A code-guaranteed-correct explanation of what this model does,
    inserted before the LLM's own narrative - the category list and
    task framing can never be misstated, unlike if the LLM had to
    describe it from scratch."""
    category_list = ", ".join(f'"{c}"' for c in categories)
    return (
        f"This model performs **text classification**: given a document, it assigns it to "
        f"exactly one of {len(categories)} categories - {category_list}. Each example below "
        f"shows a real document, the category the model predicted for it, and (highlighted "
        f"directly in the text) the specific sentences and phrases that most influenced that "
        f"prediction.\n\n---\n\n"
    )


def _render_document_block(per_doc_result: dict) -> str:
    text_html = lime_text_module.render_document_html(per_doc_result, top_k_ngrams_per_sentence=8)
    predicted_class = per_doc_result["predicted_class"]
    intended_category = per_doc_result.get("intended_category")
    probability = per_doc_result["original_probability"]

    mismatch_note = ""
    if intended_category and intended_category != predicted_class:
        mismatch_note = (
            '<div style="font-size:12px; color:#b45309; margin-bottom:4px;">'
            f'Generated as &quot;{intended_category}&quot;, but the model predicted '
            f'&quot;{predicted_class}&quot; instead.</div>'
        )

    return (
        '<div style="border:1px solid #e2e8f0; border-radius:6px; padding:12px; margin-bottom:14px;">'
        f'<div style="font-size:12px; color:#64748b; margin-bottom:6px;">'
        f'Predicted category: <strong>{predicted_class}</strong> (probability {probability:.3f})</div>'
        f"{mismatch_note}"
        f'<div style="font-size:13px; line-height:1.7;">{text_html}</div>'
        "</div>"
    )


def build_classified_documents_html(per_doc_results: list, categories: list) -> str:
    """Groups every document by the category it was generated for,
    preserving the model's declared category order, rendering a
    markdown heading per category followed by that category's
    documents."""
    by_category = {c: [] for c in categories}
    for result in per_doc_results:
        by_category.setdefault(result.get("intended_category") or "uncategorized", []).append(result)

    sections = []
    for category, results in by_category.items():
        if not results:
            continue
        sections.append(f"## {category}\n")
        sections.extend(_render_document_block(r) for r in results)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------
# LLM report body (build_header_block is shared - see ../report_header.py)
# ---------------------------------------------------------------------

def build_report_json(model_name: str, categories: list, per_doc_results: list) -> dict:
    """
    Field names are deliberately generic (no 'lime', 'occlusion',
    'n-gram', etc.) - this JSON is fed directly into the report LLM's
    prompt, so a method name that never appears here cannot leak into
    the output.
    """
    documents_generated_per_category = {c: 0 for c in categories}
    predicted_category_distribution = {}
    for r in per_doc_results:
        intended = r.get("intended_category")
        if intended in documents_generated_per_category:
            documents_generated_per_category[intended] += 1
        predicted = r.get("predicted_class")
        predicted_category_distribution[predicted] = predicted_category_distribution.get(predicted, 0) + 1

    return {
        "model_name": model_name,
        "job_type": "classification",
        "categories": categories,
        "num_documents_analyzed": len(per_doc_results),
        "documents_generated_per_category": documents_generated_per_category,
        "predicted_category_distribution": predicted_category_distribution,
    }


def generate_report_markdown(report_json: dict) -> str:
    system_prompt = (
        "You are writing a short, professional model explainability report for a business "
        "audience, for a TEXT classification model. Be factual and concise, and do not "
        "speculate beyond what the data shows. Do not write a title, a details/metadata "
        "section, or a categories list yourself - that has already been prepared separately. "
        "Do not invent category names beyond the ones given in the data.\n\n"
        "CRITICAL - methodology confidentiality: this analysis uses internal proprietary "
        "techniques. Do NOT mention, name, or describe the specific analytical methods used - "
        "never use the terms LIME, occlusion, leave-one-out, perturbation, sentence-level, "
        "n-gram, or any other specific algorithm name, anywhere in the report. Describe "
        "findings in plain business language instead.\n\n"
        "Write one short paragraph: mention how many example documents were analyzed and how "
        "they were distributed across categories, and briefly explain that each example below "
        "is shown with the specific words and sentences that most influenced its predicted "
        f"category highlighted directly in the text. Then insert the exact marker "
        f"'{DOCUMENTS_MARKER}' alone on its own line immediately after your paragraph.\n\n"
        "Close with a brief note on how these results should be used (e.g. as a starting point "
        "for review, not a final verdict)."
    )
    user_prompt = f"Write the report body based on this data:\n\n{report_json}"
    return call_llm_text(system_prompt, user_prompt)


def save_report(markdown_text: str, classified_documents_html: str, task_description_block: str,
                 header_block: str, output_path: str) -> str:
    final_text = markdown_text.replace(DOCUMENTS_MARKER, classified_documents_html or "")
    final_text = header_block + task_description_block + final_text

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(final_text)

    return output_path
