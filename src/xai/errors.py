"""
errors.py

Custom exceptions for feature_stats schema violations. Used by every
tool in xai/ (lime.py, pdp.py, ...) so that missing-schema failures are
explicit and easy to catch upstream, instead of surfacing as a generic
KeyError deep inside a perturbation loop.
"""


class FeatureSchemaError(Exception):
    """Base class for all feature_stats schema violations."""


class UnknownFeatureError(FeatureSchemaError):
    """Raised when a feature is not found in feature_stats at all."""

    def __init__(self, feature_name):
        self.feature_name = feature_name
        super().__init__(
            f"Feature '{feature_name}' was not found in feature_stats as either "
            f"a numeric or categorical feature."
        )


class MissingCategoricalCategoriesError(FeatureSchemaError):
    """
    Raised when a categorical feature has no known category list.
    This is a hard failure by design - we never guess or invent a
    category set for a field we don't have training-time evidence for.
    """

    def __init__(self, feature_name):
        self.feature_name = feature_name
        super().__init__(
            f"Categorical feature '{feature_name}' is missing a non-empty "
            f"'categories' list in feature_stats. Refusing to perturb this "
            f"feature without a known category set."
        )


class MissingNumericStatsError(FeatureSchemaError):
    """
    Raised when a numeric feature is missing 'mean' or 'std'. These two
    are mandatory for every numeric feature - there is no fallback for
    them, unlike percentiles/min/max which are optional.
    """

    def __init__(self, feature_name, missing_keys):
        self.feature_name = feature_name
        self.missing_keys = missing_keys
        super().__init__(
            f"Numeric feature '{feature_name}' is missing required stats "
            f"{missing_keys} in feature_stats. 'mean' and 'std' are mandatory "
            f"for every numeric feature."
        )