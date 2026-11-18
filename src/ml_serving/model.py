"""Load and run versioned linear model artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class InvalidFeatures(ValueError):
    """Raised when a prediction request has an invalid feature vector."""


class InvalidModelArtifact(ValueError):
    """Raised when a model artifact cannot be used by this worker."""


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A versioned linear classifier loaded from a JSON artifact."""

    model_version: str
    feature_count: int
    labels: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]

    def predict(self, features: Sequence[float]) -> tuple[str, tuple[float, ...]]:
        if len(features) != self.feature_count:
            raise InvalidFeatures(
                f"expected {self.feature_count} features, received {len(features)}"
            )

        try:
            values = tuple(float(value) for value in features)
        except (TypeError, ValueError) as error:
            raise InvalidFeatures("features must all be numbers") from error
        if not all(math.isfinite(value) for value in values):
            raise InvalidFeatures("features must all be finite numbers")

        logits = tuple(
            sum(weight * value for weight, value in zip(row[:-1], values, strict=True))
            + row[-1]
            for row in self.weights
        )
        peak = max(logits)
        exponents = tuple(math.exp(value - peak) for value in logits)
        denominator = sum(exponents)
        scores = tuple(value / denominator for value in exponents)
        best_index = max(range(len(scores)), key=scores.__getitem__)
        return self.labels[best_index], scores


def default_model_artifact_path() -> Path:
    """Return the baseline artifact shipped with the Python package."""
    return Path(__file__).parent / "artifacts" / "baseline-0.1.json"


def load_model_artifact(path: str | Path) -> ModelArtifact:
    """Read and validate a model artifact before the worker starts serving."""
    artifact_path = Path(path)
    try:
        with artifact_path.open(encoding="utf-8") as artifact_file:
            data = json.load(artifact_file)
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidModelArtifact(
            f"could not load model artifact {artifact_path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise InvalidModelArtifact("model artifact must be a JSON object")

    model_version = data.get("model_version")
    feature_count = data.get("feature_count")
    labels = data.get("labels")
    weights = data.get("weights")
    if not isinstance(model_version, str) or not model_version.strip():
        raise InvalidModelArtifact("model_version must be a non-empty string")
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count <= 0:
        raise InvalidModelArtifact("feature_count must be a positive integer")
    if not isinstance(labels, list) or not labels or not all(
        isinstance(label, str) and label.strip() for label in labels
    ):
        raise InvalidModelArtifact("labels must be a non-empty list of non-empty strings")
    if len(set(labels)) != len(labels):
        raise InvalidModelArtifact("labels must be unique")
    if not isinstance(weights, list) or len(weights) != len(labels):
        raise InvalidModelArtifact("weights must contain one row for each label")

    parsed_weights: list[tuple[float, ...]] = []
    for row in weights:
        if not isinstance(row, list) or len(row) != feature_count + 1:
            raise InvalidModelArtifact(
                "each weights row must contain one value per feature plus a bias"
            )
        try:
            parsed_row = tuple(float(value) for value in row)
        except (TypeError, ValueError) as error:
            raise InvalidModelArtifact("weights must contain numbers") from error
        if not all(math.isfinite(value) for value in parsed_row):
            raise InvalidModelArtifact("weights must all be finite numbers")
        parsed_weights.append(parsed_row)

    return ModelArtifact(
        model_version=model_version,
        feature_count=feature_count,
        labels=tuple(labels),
        weights=tuple(parsed_weights),
    )


DEFAULT_MODEL = load_model_artifact(default_model_artifact_path())
MODEL_VERSION = DEFAULT_MODEL.model_version


def predict(features: Sequence[float]) -> tuple[str, tuple[float, ...]]:
    """Predict with the packaged baseline model for backwards compatibility."""
    return DEFAULT_MODEL.predict(features)
