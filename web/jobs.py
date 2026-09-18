"""
jobs.py

Background test runs: each gets a unique session_id, its own
results/<session_id>/ directory, and a status.json updated as it
progresses. The directory is the source of truth (not an in-memory
registry) - status, results, and history all survive this process
restarting, since results/ is a mounted volume.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone

from pipeline.run_pipeline import run_pipeline

from output_handler import LocalResultsOutputHandler

logger = logging.getLogger(__name__)

RESULTS_DIR = os.environ.get("RESULTS_DIR", "/app/results")
MODEL_SERVING_URL = os.environ.get("MODEL_SERVING_URL", "http://model_serving:8000")

# Text models cost far more per point (hierarchical occlusion per
# document - roughly one predict call per sentence plus
# one per n-gram in the top sentence) than tabular ones (one LIME
# sample = one predict call) - see
# src/pipeline/tests/test_run_pipeline.py's own note - so this demo
# uses a smaller default for them, to keep a demo run's wait time
# reasonable.
NUM_POINTS_BY_DATASET = {"bbc_news": 20}
DEFAULT_NUM_POINTS = 100
DEFAULT_NUM_LIME_SAMPLES = 500


def _status_path(session_dir: str) -> str:
    return os.path.join(session_dir, "status.json")


def _write_status(session_dir: str, **fields) -> None:
    path = _status_path(session_dir)
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data.update(fields)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_test_run(dataset: str, model_key: str, num_points: int = None) -> str:
    session_id = uuid.uuid4().hex
    session_dir = os.path.join(RESULTS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    _write_status(
        session_dir,
        session_id=session_id, dataset=dataset, model_key=model_key,
        status="pending", created_at=_now(),
    )

    thread = threading.Thread(target=_run, args=(session_id, session_dir, dataset, model_key, num_points), daemon=True)
    thread.start()

    return session_id


def _run(session_id: str, session_dir: str, dataset: str, model_key: str, num_points: int = None) -> None:
    if num_points is None:
        num_points = NUM_POINTS_BY_DATASET.get(dataset, DEFAULT_NUM_POINTS)
    _write_status(session_dir, status="running", started_at=_now(), num_points_requested=num_points)
    try:
        result = run_pipeline(
            base_url=f"{MODEL_SERVING_URL}/{dataset}",
            model_key=model_key,
            output_handler=LocalResultsOutputHandler(session_dir),
            num_points=num_points,
            num_lime_samples=DEFAULT_NUM_LIME_SAMPLES,
        )
        with open(os.path.join(session_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        _write_status(session_dir, status="completed", completed_at=_now())
    except Exception as e:
        logger.exception("Test run %s failed", session_id)
        _write_status(session_dir, status="failed", error=str(e), completed_at=_now())


def get_status(session_id: str):
    path = _status_path(os.path.join(RESULTS_DIR, session_id))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_result(session_id: str):
    path = os.path.join(RESULTS_DIR, session_id, "result.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_artifact_path(session_id: str, artifact_key: str):
    session_dir = os.path.join(RESULTS_DIR, session_id)
    manifest_path = os.path.join(session_dir, "artifacts.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    filename = manifest.get(artifact_key)
    if not filename:
        return None
    full_path = os.path.join(session_dir, filename)
    return full_path if os.path.exists(full_path) else None


def get_report_json(session_id: str):
    path = get_artifact_path(session_id, "report_json")
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def list_sessions() -> list:
    if not os.path.isdir(RESULTS_DIR):
        return []
    sessions = []
    for session_id in os.listdir(RESULTS_DIR):
        status = get_status(session_id)
        if status:
            sessions.append(status)
    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return sessions
