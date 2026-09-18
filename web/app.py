"""
app.py

The 'web' core/testing service - the single API surface the UI talks
to. Proxies the model catalog and live predictions to model_serving,
and runs the explainability pipeline (src/pipeline, importable via
PYTHONPATH=/app/src - see Dockerfile) as background jobs, persisting
every session under RESULTS_DIR (a mounted volume) so status and past
results survive this process restarting.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import logging

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import jobs
from catalog import build_catalog, MODEL_SERVING_URL

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

app = FastAPI(title="Explainability Demo - Core Service")

# Permissive on purpose - this is a local demo tool, and the browser
# reaches this service from a different origin (the ui container's
# nginx, on a different port) than this API's own.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def api_models():
    """The Models tab's entire payload: every dataset -> model_key ->
    /describe response (schema, metrics, example request) from
    model_serving."""
    return build_catalog()


@app.post("/api/models/{dataset}/{model_key}/predict")
def api_predict(dataset: str, model_key: str, payload: dict):
    """The Models tab's 'Try it' button - proxies straight to
    model_serving, so the UI never needs to know model_serving exists."""
    try:
        r = requests.post(f"{MODEL_SERVING_URL}/{dataset}/{model_key}/predict", json=payload, timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"model_serving unreachable: {e}")

    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail=detail)

    return r.json()


MIN_NUM_POINTS = 1
MAX_NUM_POINTS = 500


@app.post("/api/tests")
def api_start_test(payload: dict):
    """Run Test tab's 'Run Test' button - kicks off a background job,
    returns immediately with the session_id to poll."""
    dataset = payload.get("dataset")
    model_key = payload.get("model_key")
    if not dataset or not model_key:
        raise HTTPException(status_code=400, detail="Both 'dataset' and 'model_key' are required.")

    num_points = payload.get("num_points")
    if num_points is not None:
        try:
            num_points = int(num_points)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="'num_points' must be an integer.")
        if not (MIN_NUM_POINTS <= num_points <= MAX_NUM_POINTS):
            raise HTTPException(
                status_code=400,
                detail=f"'num_points' must be between {MIN_NUM_POINTS} and {MAX_NUM_POINTS}.",
            )

    session_id = jobs.start_test_run(dataset, model_key, num_points=num_points)
    return {"session_id": session_id}


@app.get("/api/tests")
def api_list_tests():
    """Run Test tab's 'Past Runs' history list."""
    return jobs.list_sessions()


@app.get("/api/tests/{session_id}/status")
def api_test_status(session_id: str):
    status = jobs.get_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'")
    return status


@app.get("/api/tests/{session_id}/dashboard")
def api_test_dashboard(session_id: str):
    status = jobs.get_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'")
    if status.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Session '{session_id}' is not completed yet (status: {status.get('status')}).",
        )

    return {
        "status": status,
        "result": jobs.get_result(session_id) or {},
        "report": jobs.get_report_json(session_id) or {},
    }


@app.get("/api/tests/{session_id}/artifacts/{artifact_key}")
def api_test_artifact(session_id: str, artifact_key: str):
    path = jobs.get_artifact_path(session_id, artifact_key)
    if not path:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_key}' not found for session '{session_id}'")
    return FileResponse(path)
