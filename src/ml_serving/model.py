"""Small deterministic baseline used to exercise the serving path."""

from __future__ import annotations

import math
from collections.abc import Sequence

MODEL_VERSION = "baseline-0.1"
FEATURE_COUNT = 2
LABELS = ("class_a", "class_b", "class_c")

# Each row is a class's linear weights followed by its bias.
_WEIGHTS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-0.5, -0.5, 0.0))


class InvalidFeatures(ValueError):
    """Raised when a prediction request has an invalid feature vector."""


def predict(features: Sequence[float]) -> tuple[str, tuple[float, ...]]:
    """Return a label and softmax scores for exactly two finite input features."""
    if len(features) != FEATURE_COUNT:
        raise InvalidFeatures(f"expected {FEATURE_COUNT} features, received {len(features)}")

    values = tuple(float(value) for value in features)
    if not all(math.isfinite(value) for value in values):
        raise InvalidFeatures("features must all be finite numbers")

    logits = tuple(row[0] * values[0] + row[1] * values[1] + row[2] for row in _WEIGHTS)
    peak = max(logits)
    exponents = tuple(math.exp(value - peak) for value in logits)
    denominator = sum(exponents)
    scores = tuple(value / denominator for value in exponents)
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return LABELS[best_index], scores
