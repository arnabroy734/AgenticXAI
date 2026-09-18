"""
output_handler.py

Abstract persistence boundary between "what artifacts a pipeline run
produced" (fixed, model-agnostic - the same shape regardless of which
modality backend generated them) and "where they end up" (local disk
for testing, a remote POST/object-store call in production). Neither
concrete implementation lives here - the caller (e.g. a microservice)
supplies one.
"""

from abc import ABC, abstractmethod


class OutputHandler(ABC):
    @abstractmethod
    def persist(self, working_dir: str, artifact_filenames: dict) -> dict:
        """
        working_dir: local directory where run_pipeline wrote every
            artifact for this run (charts, report json/markdown/pdf).
            Still exists (not yet cleaned up) when this is called.
        artifact_filenames: {artifact_key: filename_within_working_dir}.
            Keys are generic - "report_pdf", "report_json",
            "report_markdown", "sensitivity_chart",
            "impact_chart_<feature_name>", "sample_case_<n>" - never
            backend-specific step names, so any modality backend
            (tabular today, text or anything else later) populates the
            same shape and this contract never needs to change.

        Returns {artifact_key: final_path_or_url} - wherever this
        handler decided each artifact now lives (a permanent local
        path, a remote URL, a storage key, etc). This becomes the
        "artifacts" field of run_pipeline's return value.
        """
        raise NotImplementedError
