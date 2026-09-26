"""Calibration metrics shared by the rediscovery benchmark and the clinical arm.

Ref: Sec. 4.4 and Sec. 4.6 (expected calibration error and Brier score are defined
identically at discovery time and at clinical time so the two are comparable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BINS = 10


class CalibrationMetricError(ValueError):
    """Raised when a calibration metric is asked for inconsistent inputs."""


def _validate(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(probabilities, dtype=float)
    observed = np.asarray(labels, dtype=float)
    if predicted.shape != observed.shape:
        raise CalibrationMetricError("probabilities and labels must have the same shape")
    if predicted.size == 0:
        raise CalibrationMetricError("calibration metrics need at least one observation")
    if np.any(predicted < -1e-9) or np.any(predicted > 1.0 + 1e-9):
        raise CalibrationMetricError("probabilities must lie in [0, 1]")
    return predicted, observed


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    predicted, observed = _validate(probabilities, labels)
    return float(np.mean((predicted - observed) ** 2))


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = DEFAULT_BINS
) -> float:
    predicted, observed = _validate(probabilities, labels)
    if bins < 1:
        raise CalibrationMetricError(f"bin count must be positive: {bins!r}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (predicted >= lower) & (predicted <= upper)
        else:
            mask = (predicted >= lower) & (predicted < upper)
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        confidence = float(np.mean(predicted[mask]))
        accuracy = float(np.mean(observed[mask]))
        total += (count / predicted.size) * abs(confidence - accuracy)
    return float(total)


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    mean_accuracy: float

    @property
    def gap(self) -> float:
        return self.mean_confidence - self.mean_accuracy

    def as_dict(self) -> dict[str, float | int]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "mean_accuracy": self.mean_accuracy,
            "gap": self.gap,
        }


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = DEFAULT_BINS
) -> tuple[ReliabilityBin, ...]:
    predicted, observed = _validate(probabilities, labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    curve: list[ReliabilityBin] = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == bins - 1:
            mask = (predicted >= lower) & (predicted <= upper)
        else:
            mask = (predicted >= lower) & (predicted < upper)
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        curve.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=float(np.mean(predicted[mask])),
                mean_accuracy=float(np.mean(observed[mask])),
            )
        )
    return tuple(curve)


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    brier: float
    expected_calibration_error: float
    bins: int
    observations: int
    mean_probability: float
    base_rate: float

    @property
    def bias(self) -> float:
        return self.mean_probability - self.base_rate

    def as_dict(self) -> dict[str, float | int]:
        return {
            "brier": self.brier,
            "expected_calibration_error": self.expected_calibration_error,
            "bins": self.bins,
            "observations": self.observations,
            "mean_probability": self.mean_probability,
            "base_rate": self.base_rate,
            "bias": self.bias,
        }


def calibration_summary(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = DEFAULT_BINS
) -> CalibrationSummary:
    predicted, observed = _validate(probabilities, labels)
    return CalibrationSummary(
        brier=brier_score(predicted, observed),
        expected_calibration_error=expected_calibration_error(predicted, observed, bins=bins),
        bins=bins,
        observations=int(predicted.size),
        mean_probability=float(np.mean(predicted)),
        base_rate=float(np.mean(observed)),
    )


def overconfidence(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Signed confidence minus accuracy pooled over all observations."""

    predicted, observed = _validate(probabilities, labels)
    return float(np.mean(predicted) - np.mean(observed))
