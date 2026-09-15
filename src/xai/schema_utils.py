"""
schema_utils.py

Shared validation and resolution helpers for feature_stats, used by every
tool in xai/. Encodes two hard rules:

  1. Categorical features MUST have a non-empty 'categories' list.
     There is no fallback - if it's missing, we raise, we don't guess.

  2. Numeric features MUST have 'mean' and 'std'. These are mandatory.
     Everything else (p25, p75, min, max) is optional. When the optional
     stats are missing, range-based tools (like PDP) fall back to
     sampling around mean +/- a scaled std, instead of failing.
"""

from .errors import (
    UnknownFeatureError,
    MissingCategoricalCategoriesError,
    MissingNumericStatsError,
)



def validate_feature_stats(feature_stats: dict, feature_names: list) -> None:
    """
    Validates that every feature in feature_names has usable stats in
    feature_stats. Raises immediately on the first violation found.

    feature_stats is expected to have the same shape as what's saved in
    checkpoints/<model>/feature_stats.json:
        {
            "numeric_features": {"<name>": {"mean": ..., "std": ..., ...}, ...},
            "categorical_features": {"<name>": {"categories": [...], "frequencies": {...}}, ...},
            ...
        }
    """
    numeric_features = feature_stats.get("numeric_features", {})
    categorical_features = feature_stats.get("categorical_features", {})

    for name in feature_names:
        if name in numeric_features:
            _validate_numeric_stats(name, numeric_features[name])
        elif name in categorical_features:
            _validate_categorical_stats(name, categorical_features[name])
        else:
            raise UnknownFeatureError(name)


def _validate_numeric_stats(feature_name: str, stats: dict) -> None:
    missing = [k for k in ("mean", "std") if stats.get(k) is None]
    if missing:
        raise MissingNumericStatsError(feature_name, missing)


def _validate_categorical_stats(feature_name: str, stats: dict) -> None:
    categories = stats.get("categories")
    if not categories:
        raise MissingCategoricalCategoriesError(feature_name)


def get_numeric_perturbation_std(stats: dict, scale: float = 1.0) -> float:
    """
    Returns the std to use for Gaussian perturbation around an instance's
    value (used by LIME). 'std' is mandatory, so this never falls back -
    validate_feature_stats() should already have been called first.
    """
    return stats["std"] * scale


def get_numeric_range(stats: dict, fallback_scale: float = 2.0) -> tuple:
    """
    Returns (low, high) for grid-based sampling (used by PDP). Preference
    order:
        1. (p25, p75) if both present
        2. (min, max) if both present
        3. (mean - fallback_scale*std, mean + fallback_scale*std)

    Only reached after validate_feature_stats() has confirmed 'mean' and
    'std' exist, so the final fallback is always available.
    """
    p25, p75 = stats.get("p25"), stats.get("p75")
    if p25 is not None and p75 is not None:
        return p25, p75

    lo, hi = stats.get("min"), stats.get("max")
    if lo is not None and hi is not None:
        return lo, hi

    mean, std = stats["mean"], stats["std"]
    return mean - fallback_scale * std, mean + fallback_scale * std