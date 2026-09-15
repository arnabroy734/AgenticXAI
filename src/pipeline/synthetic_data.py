"""
synthetic_data.py

Step 2 + 3: calls the LLM to generate synthetic raw feature rows
matching feature_stats, validates every row's schema and bounds, and
retries the WHOLE batch up to max_retries times if any row fails.

Row validation is intentionally strict and separate from
xai/schema_utils.py - that module validates the shape of feature_stats
itself (does every feature have the required stats), while this module
validates individual candidate DATA rows against those stats.
"""

from .llm_client import call_llm_json


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

        print(f"Attempt {attempt}/{max_retries}: requesting {remaining} more row(s) ...")
        candidate_rows = _call_llm_for_rows(feature_stats, remaining)

        attempt_valid = []
        attempt_invalid = 0
        for row in candidate_rows:
            if validate_row(row, feature_stats):
                attempt_invalid += 1
            else:
                attempt_valid.append(row)

        valid_rows.extend(attempt_valid)
        print(
            f"  Got {len(attempt_valid)} valid / {attempt_invalid} invalid row(s). "
            f"Total valid so far: {len(valid_rows)}/{num_points}"
        )

    if len(valid_rows) < num_points:
        print(
            f"WARNING: only {len(valid_rows)}/{num_points} synthetic rows passed validation "
            f"after {max_retries} attempts. Proceeding with the {len(valid_rows)} valid row(s) "
            f"collected instead of discarding them."
        )

    if not valid_rows:
        raise RuntimeError(
            f"No valid synthetic rows could be generated after {max_retries} attempts. "
            f"Cannot proceed with zero data points."
        )

    return valid_rows