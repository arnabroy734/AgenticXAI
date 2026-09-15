"""
predict_client.py

Step 4: a single predict_fn factory that works identically for
regression and classification models, since both return one scalar
value under a known field name (learned from probe_model's output_field,
sourced from the model's own /describe contract).

Also handles a real, recurring mismatch: LIME/PDP perturb all numeric
fields continuously, but a served model's schema may require some
numeric fields to be strict integers (e.g. real counts like bedrooms),
and may also enforce range constraints (e.g. a field must be >= 0).
There is no reliable way to predict either of these in advance from
feature_stats alone (a field's observed min/max can coincidentally be
whole numbers even when genuinely continuous - see area's min/max in
the house price dataset). Instead, this reacts to the API's own
validation response: on a 422, it inspects every reported violation and
attempts a targeted fix (round for int-type errors, clip for range
errors), retrying up to 2 times. If it still fails, it raises a
detailed error showing exactly which fields and constraints were
violated and the payload that triggered it - never a vague, opaque
HTTP error - so a persistent failure is immediately diagnosable.
"""

import json as json_module

import requests


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


def make_predict_fn(predict_url: str, output_field: str, feature_stats: dict = None, max_retries: int = 2):
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
        return float(r.json()[output_field])

    return predict_fn