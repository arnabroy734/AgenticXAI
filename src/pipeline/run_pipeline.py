"""
run_pipeline.py

Functional entry point for the explainability reporting pipeline -
designed to be imported and called by a microservice, not run as a
CLI: every parameter is explicit, including how generated artifacts
are finally persisted (via an injected OutputHandler). No argv
parsing lives here.

Dispatches to a modality-specific backend based on the probed model's
/describe response ("modality", defaulting to "tabular" if absent).
Adding a genuinely new model modality later means writing one new
backend module with a matching run(...) contract (see
pipeline/tabular/backend.py) and adding one line to BACKENDS below -
not editing this dispatch logic.
"""

import logging
import re
import shutil
import tempfile

import requests

from .output_handler import OutputHandler
from .tabular import backend as tabular_backend
from .text import backend as text_backend

logger = logging.getLogger(__name__)

BACKENDS = {
    "tabular": tabular_backend.run,
    "text": text_backend.run,
}


def _sanitize_for_path(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def run_pipeline(
    base_url: str,
    model_key: str,
    output_handler: OutputHandler,
    num_points: int = 100,
    num_lime_samples: int = 500,
) -> dict:
    """
    Returns a model-agnostic result dict:
        {
            "model_key": str,
            "model_name": str,
            "job_type": "regression" or "classification",
            "reference_class": str or None,       # classification only
            "num_points_analyzed": int,
            "artifacts": {...whatever output_handler.persist(...) returned...},
        }

    Raises whatever the probe/backend/output_handler raise (e.g. a
    ValueError for a classification model missing output.reference_class,
    a RuntimeError if zero synthetic rows validated, a
    pipeline.tabular.model_client.PredictionRequestError if the served
    model keeps rejecting requests) - the caller decides how to map
    those to a response, this function doesn't catch or reframe them.
    """
    describe_url = f"{base_url}/{model_key}/describe"
    logger.info("Fetching %s", describe_url)
    r = requests.get(describe_url)
    r.raise_for_status()
    describe_body = r.json()

    modality = describe_body.get("modality", "tabular")
    backend_run = BACKENDS.get(modality)
    if backend_run is None:
        raise ValueError(
            f"No pipeline backend registered for modality '{modality}' (known: {sorted(BACKENDS)})."
        )

    working_dir = tempfile.mkdtemp(
        prefix=f"xai_{_sanitize_for_path(base_url)}_{_sanitize_for_path(model_key)}_"
    )
    logger.info("Working directory: %s", working_dir)

    try:
        backend_result = backend_run(
            base_url=base_url,
            model_key=model_key,
            describe_body=describe_body,
            num_points=num_points,
            num_lime_samples=num_lime_samples,
            working_dir=working_dir,
        )

        logger.info("Persisting artifacts via %s", type(output_handler).__name__)
        artifacts = output_handler.persist(working_dir, backend_result["artifact_filenames"])

        return {
            "model_key": model_key,
            "model_name": backend_result["model_name"],
            "job_type": backend_result["job_type"],
            "reference_class": backend_result.get("reference_class"),
            "num_points_analyzed": backend_result["num_points_analyzed"],
            "artifacts": artifacts,
        }
    finally:
        shutil.rmtree(working_dir, ignore_errors=True)
