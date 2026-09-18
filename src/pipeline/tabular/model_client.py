"""
model_client.py

Everything about talking to a served tabular model:
  - Step 1: parse an already-fetched /describe response to learn job
    type, feature schema, which field to read from /predict responses,
    and (for classification) which class's probability is the fixed
    explainability target. The HTTP GET itself happens once, in
    run_pipeline.py, before it even knows this is a tabular model - it
    only needs job_type/modality to pick a backend - so this module
    just parses the body it's handed, rather than fetching it again.
  - Step 4: build a predict_fn that works identically for regression
    and classification, self-correcting on the API's own validation
    errors.
  - Step 2-3: generate + validate synthetic raw feature rows via LLM.
"""

import json as json_module
import logging

import requests

from ..llm_client import call_llm_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Step 1: parse /describe
# ---------------------------------------------------------------------

def parse_describe_body(describe_body: dict, base_url: str, model_key: str) -> dict:
    """
    Returns:
        {
            "job_type": "regression" or "classification",
            "feature_stats": {"numeric_features": {...}, "categorical_features": {...}},
            "output_field": "predicted_price",   # field name to read from /predict responses
            "reference_class": "yes" or None,     # classification only: which class's
                                                     # probability output_field's value refers to
            "predict_url": "<base_url>/<model_key>/predict",
        }

    For classification jobs, the API's /describe response must declare
    output.reference_class - the one class whose probability stays
    fixed as the explainability target throughout LIME/PDP, regardless
    of which class the model itself would predict for any given
    perturbed sample. Without a fixed target, attributions would track
    a moving target (whichever class currently wins the argmax) and
    produce discontinuous, misleading explanations. This is a hard
    failure, not a guess - the API is the single source of truth for
    which class that is.
    """
    numeric_features, categorical_features = {}, {}
    for f in describe_body["features"]:
        if f["field_type"] == "numeric":
            numeric_features[f["name"]] = f["stats"]
        else:
            categorical_features[f["name"]] = {
                "categories": f["categories"],
                "frequencies": f["frequencies"],
            }

    feature_stats = {"numeric_features": numeric_features, "categorical_features": categorical_features}
    predict_url = f"{base_url}/{model_key}/predict"

    job_type = describe_body["job_type"]
    output = describe_body["output"]

    reference_class = None
    if job_type == "classification":
        reference_class = output.get("reference_class")
        if not reference_class:
            raise ValueError(
                f"Classification model '{model_key}' did not declare 'output.reference_class' "
                f"in its /describe response - cannot determine which class's probability to explain."
            )

    return {
        "job_type": job_type,
        "model_name": describe_body["model_name"],
        "feature_stats": feature_stats,
        "output_field": output["field"],
        "reference_class": reference_class,
        "predict_url": predict_url,
    }


# ---------------------------------------------------------------------
# Step 4: predict_fn factory
# ---------------------------------------------------------------------

class PredictionRequestError(RuntimeError):
    """Raised when the API rejects a request even after automatic correction
    attempts. Carries the full validation detail and the final payload for
    diagnosis, instead of surfacing a generic, opaque HTTP error."""

    def __init__(self, violations, payload):
        self.violations = violations
        self.payload = payload
        super().__init__(
            f"Prediction request rejected after automatic correction attempts.\n"
            f"Violations: {json_module.dumps(violations, indent=2)}\n"
            f"Final payload sent: {json_module.dumps(payload, indent=2)}"
        )


def _get_violations(response) -> list:
    try:
        return response.json().get("detail", [])
    except ValueError:
        return []


def _apply_fix(payload: dict, violation: dict) -> bool:
    """Attempts to fix a single validation violation in-place. Returns
    True if a fix was applied, False if this violation type isn't
    something we know how to correct automatically."""
    loc = violation.get("loc", [])
    if not loc:
        return False
    field = loc[-1]
    if field not in payload:
        return False

    error_type = violation.get("type")

    if error_type == "int_from_float":
        payload[field] = int(round(payload[field]))
        return True

    if error_type in ("greater_than_equal", "greater_than"):
        bound = violation.get("ctx", {}).get(
            "ge" if error_type == "greater_than_equal" else "gt"
        )
        if bound is not None:
            payload[field] = bound if error_type == "greater_than_equal" else bound + 1e-9
            return True

    if error_type in ("less_than_equal", "less_than"):
        bound = violation.get("ctx", {}).get(
            "le" if error_type == "less_than_equal" else "lt"
        )
        if bound is not None:
            payload[field] = bound if error_type == "less_than_equal" else bound - 1e-9
            return True

    return False


def make_predict_fn(
    predict_url: str,
    output_field: str,
    feature_stats: dict = None,
    job_type: str = "regression",
    reference_class: str = None,
    max_retries: int = 2,
):
    """
    For regression, reads output_field as a flat scalar. For
    classification, output_field is a dict of per-class probabilities
    and reference_class picks out the one fixed class LIME/PDP explain
    (see probe_model's docstring for why this must be fixed).
    """
    def predict_fn(raw_dict: dict) -> float:
        payload = dict(raw_dict)

        for attempt in range(max_retries + 1):
            r = requests.post(predict_url, json=payload)

            if r.status_code != 422:
                break

            violations = _get_violations(r)
            any_fixed = False
            for violation in violations:
                if _apply_fix(payload, violation):
                    any_fixed = True

            if not any_fixed:
                raise PredictionRequestError(violations, payload)

        if r.status_code == 422:
            raise PredictionRequestError(_get_violations(r), payload)

        r.raise_for_status()
        output_value = r.json()[output_field]

        if job_type == "classification":
            return float(output_value[reference_class])
        return float(output_value)

    return predict_fn


# ---------------------------------------------------------------------
# Step 2-3: synthetic data generation + validation
#
# Row validation is intentionally strict and separate from
# xai/schema_utils.py - that module validates the shape of feature_stats
# itself (does every feature have the required stats), while this
# validates individual candidate DATA rows against those stats.
# ---------------------------------------------------------------------

def validate_row(row: dict, feature_stats: dict) -> list:
    """Returns a list of human-readable violation strings; empty list means valid."""
    violations = []

    for name, stats in feature_stats["numeric_features"].items():
        if name not in row:
            violations.append(f"missing numeric field '{name}'")
            continue
        value = row[name]
        if not isinstance(value, (int, float)):
            violations.append(f"'{name}'={value!r} is not numeric")
            continue
        lo = stats.get("min", stats["mean"] - 4 * stats["std"])
        hi = stats.get("max", stats["mean"] + 4 * stats["std"])
        if not (lo <= value <= hi):
            violations.append(f"'{name}'={value} out of bounds [{lo}, {hi}]")

    for name, stats in feature_stats["categorical_features"].items():
        if name not in row:
            violations.append(f"missing categorical field '{name}'")
            continue
        if row[name] not in stats["categories"]:
            violations.append(f"'{name}'={row[name]!r} not in {stats['categories']}")

    return violations


def _build_prompt(feature_stats: dict, num_points: int) -> tuple:
    numeric_desc = []
    for name, stats in feature_stats["numeric_features"].items():
        lo = stats.get("min")
        hi = stats.get("max")
        numeric_desc.append(f"- {name}: numeric, must stay within [{lo}, {hi}], training mean={stats['mean']:.2f}")

    categorical_desc = []
    for name, stats in feature_stats["categorical_features"].items():
        categorical_desc.append(f"- {name}: must be exactly one of {stats['categories']}")

    system_prompt = (
        "You are a synthetic tabular data generator. You produce realistic, diverse data "
        "points strictly within the given schema bounds. Respond ONLY with a JSON object "
        'of the form {"rows": [ {...}, {...}, ... ]} with no extra commentary, no markdown '
        "fences, and no fields other than the ones listed."
    )

    user_prompt = (
        f"Generate exactly {num_points} synthetic data points as a JSON object with a "
        f'single key "rows" containing a list of {num_points} objects.\n\n'
        f"Numeric fields (must stay within the given bounds):\n" + "\n".join(numeric_desc) + "\n\n"
        f"Categorical fields (must use exactly one of the listed values):\n" + "\n".join(categorical_desc) + "\n\n"
        f"Every row must include every field listed above, with realistic, varied, "
        f"non-repetitive combinations."
    )

    return system_prompt, user_prompt


def _call_llm_for_rows(feature_stats: dict, num_points: int) -> list:
    system_prompt, user_prompt = _build_prompt(feature_stats, num_points)
    result = call_llm_json(system_prompt, user_prompt)
    return result.get("rows", [])


def generate_synthetic_data(feature_stats: dict, num_points: int = 100, max_retries: int = 3) -> list:
    """
    Calls the LLM to generate num_points synthetic raw rows matching
    feature_stats. Only individually valid rows are kept - an invalid
    row is dropped, not the whole batch. Each retry asks only for the
    remaining shortfall, topping up the valid set.

    If fewer than num_points valid rows remain after max_retries
    attempts, this does NOT discard what was generated and does NOT
    raise - it proceeds with whatever valid rows were collected (with
    a warning printed). It only raises if zero valid rows could be
    produced at all, since there would be nothing to proceed with.
    """
    valid_rows = []

    for attempt in range(1, max_retries + 1):
        remaining = num_points - len(valid_rows)
        if remaining <= 0:
            break

        logger.info("Attempt %d/%d: requesting %d more row(s)", attempt, max_retries, remaining)
        candidate_rows = _call_llm_for_rows(feature_stats, remaining)

        attempt_valid = []
        attempt_invalid = 0
        for row in candidate_rows:
            if validate_row(row, feature_stats):
                attempt_invalid += 1
            else:
                attempt_valid.append(row)

        valid_rows.extend(attempt_valid)
        logger.info(
            "Got %d valid / %d invalid row(s). Total valid so far: %d/%d",
            len(attempt_valid), attempt_invalid, len(valid_rows), num_points,
        )

    if len(valid_rows) < num_points:
        logger.warning(
            "Only %d/%d synthetic rows passed validation after %d attempts. "
            "Proceeding with the %d valid row(s) collected instead of discarding them.",
            len(valid_rows), num_points, max_retries, len(valid_rows),
        )

    if not valid_rows:
        raise RuntimeError(
            f"No valid synthetic rows could be generated after {max_retries} attempts. "
            f"Cannot proceed with zero data points."
        )

    return valid_rows
