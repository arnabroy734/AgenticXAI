"""
report_header.py

Deterministic report header block, shared by every modality backend's
report module - none of it (model tested, endpoint, task type,
reference_class, data points analyzed, timestamp) depends on modality,
so it lives once here rather than being duplicated per backend.
Generated directly by code, never by the LLM, so these facts can never
be misstated.
"""

from datetime import datetime
from zoneinfo import ZoneInfo


def build_header_block(base_url: str, model_name: str, job_type: str, num_points: int,
                        reference_class: str = None) -> str:
    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")

    task_type_display = job_type
    if job_type == "classification" and reference_class:
        task_type_display = f"{job_type} (explaining likelihood of: {reference_class})"

    return (
        "# Model Explainability Report\n\n"
        "| | |\n"
        "|---|---|\n"
        f"| **Model tested** | `{model_name}` |\n"
        f"| **Endpoint** | `{base_url}` |\n"
        f"| **Task type** | {task_type_display} |\n"
        f"| **Data points analyzed** | {num_points} |\n"
        f"| **Report generated** | {timestamp} |\n\n"
        "---\n\n"
    )
