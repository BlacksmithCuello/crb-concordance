"""DeLong inference for one ROC curve and for two correlated curves.

Ref: Sec. 4.6 (95% intervals for areas under the curve use bootstrap resampling and
the difference between two correlated receiver-operating-characteristic curves on
the same cohort is assessed by DeLong's method).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.utils.numerics import normal_ppf


class DeLongError(ValueError):
    """Raised when the placement values cannot be formed."""


@dataclass(frozen=True, slots=True)
class RocEstimate:
    """An AUC with its DeLong variance and a normal-approximation interval."""

    auc: float
    variance: float
    positives: int
    negatives: int

    @property
    def standard_error(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))

    def interval(self, *, level: float = 0.95) -> tuple[float, float]:
        critical = normal_ppf(1.0 - (1.0 - level) / 2.0)
        half = critical * self.standard_error
        return (self.auc - half, self.auc + half)

    def as_dict(self) -> dict[str, float | int]:
        lower, upper = self.interval()
        return {
            "auc": self.auc,
            "variance": self.variance,
            "standard_error": self.standard_error,
            "ci_lower": lower,
            "ci_upper": upper,
            "positives": self.positives,
            "negatives": self.negatives,
        }


@dataclass(frozen=True, slots=True)
class PairedRocComparison:
    """The DeLong test for two curves scored on the same cases."""

    first: RocEstimate
    second: RocEstimate
    covariance: float
    z: float
    p_value: float

    @property
    def difference(self) -> float:
        return self.first.auc - self.second.auc

    def interval(self, *, level: float = 0.95) -> tuple[float, float]:
        variance = self.first.variance + self.second.variance - 2.0 * self.covariance
        half = normal_ppf(1.0 - (1.0 - level) / 2.0) * float(np.sqrt(max(variance, 0.0)))
        return (self.difference - half, self.difference + half)

    def as_dict(self) -> dict[str, object]:
        lower, upper = self.interval()
        return {
            "first": self.first.as_dict(),
            "second": self.second.as_dict(),
            "difference": self.difference,
            "ci_lower": lower,
            "ci_upper": upper,
            "covariance": self.covariance,
            "z": self.z,
            "p_value": self.p_value,
        }


def _structural_components(scores: np.ndarray, positives: np.ndarray) -> np.ndarray:
    positive_scores = scores[positives]
    negative_scores = scores[~positives]
    if positive_scores.size == 0 or negative_scores.size == 0:
        raise DeLongError("both classes must be present")
    comparisons = (positive_scores[:, None] > negative_scores[None, :]).astype(float)
    ties = (positive_scores[:, None] == negative_scores[None, :]).astype(float)
    return comparisons + 0.5 * ties


def midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    position = 0
    while position < values.size:
        end = position
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[position]:
            end += 1
        average = 0.5 * (position + end) + 1.0
        ranks[order[position : end + 1]] = average
        position = end + 1
    return ranks


def roc_estimate(scores: np.ndarray, labels: np.ndarray) -> RocEstimate:
    """AUC and DeLong variance for one curve."""

    values = np.asarray(scores, dtype=float)
    positives = np.asarray(labels, dtype=bool)
    if values.shape != positives.shape:
        raise DeLongError("scores and labels must align")
    components = _structural_components(values, positives)
    n_positive = int(np.count_nonzero(positives))
    n_negative = values.size - n_positive
    auc = float(components.mean())
    v10 = components.mean(axis=1)
    v01 = components.mean(axis=0)
    variance = float(np.var(v10, ddof=1) / n_positive + np.var(v01, ddof=1) / n_negative)
    return RocEstimate(auc=auc, variance=variance, positives=n_positive, negatives=n_negative)


def paired_comparison(
    first_scores: np.ndarray, second_scores: np.ndarray, labels: np.ndarray
) -> PairedRocComparison:
    """DeLong comparison of two curves on the same cases."""

    positives = np.asarray(labels, dtype=bool)
    first = np.asarray(first_scores, dtype=float)
    second = np.asarray(second_scores, dtype=float)
    if first.shape != second.shape or first.shape != positives.shape:
        raise DeLongError("both curves and the labels must align")
    components_first = _structural_components(first, positives)
    components_second = _structural_components(second, positives)
    n_positive = int(np.count_nonzero(positives))
    n_negative = first.size - n_positive
    estimate_first = roc_estimate(first, labels)
    estimate_second = roc_estimate(second, labels)
    v10_first = components_first.mean(axis=1)
    v10_second = components_second.mean(axis=1)
    v01_first = components_first.mean(axis=0)
    v01_second = components_second.mean(axis=0)
    covariance = float(
        np.cov(v10_first, v10_second, ddof=1)[0, 1] / n_positive
        + np.cov(v01_first, v01_second, ddof=1)[0, 1] / n_negative
    )
    variance = estimate_first.variance + estimate_second.variance - 2.0 * covariance
    difference = estimate_first.auc - estimate_second.auc
    if variance <= 0.0:
        return PairedRocComparison(
            first=estimate_first,
            second=estimate_second,
            covariance=covariance,
            z=0.0,
            p_value=1.0,
        )
    z = float(difference / np.sqrt(variance))
    return PairedRocComparison(
        first=estimate_first,
        second=estimate_second,
        covariance=covariance,
        z=z,
        p_value=two_sided_normal_p(z),
    )


def two_sided_normal_p(z: float) -> float:
    from math import erfc

    return float(erfc(abs(z) / float(np.sqrt(2.0))))
