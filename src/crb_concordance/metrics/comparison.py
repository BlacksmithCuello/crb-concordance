"""Resampling and agreement statistics used by every reported comparison.

Ref: Sec. 4.6 (95% confidence intervals use bootstrap resampling with 2,000
resamples for clinical quantities and 5,000 for seed-level quantities; the
difference between two correlated AUROCs is assessed by DeLong's method; the
reader-study gain is cross-checked by a reader-level cluster bootstrap).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from crb_concordance.utils.numerics import bootstrap_indices, percentile_interval, rng_from

CLINICAL_RESAMPLES = 2000
SEED_RESAMPLES = 5000


class ComparisonError(ValueError):
    """Raised when a comparison is asked for inconsistent inputs."""


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    lower: float
    upper: float
    level: float
    resamples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
            "resamples": self.resamples,
        }

    @property
    def width(self) -> float:
        return self.upper - self.lower


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    resamples: int = CLINICAL_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        raise ComparisonError("cannot bootstrap an empty sample")
    generator = rng_from(seed)
    draws = sample[bootstrap_indices(sample.size, resamples, generator)]
    means = draws.mean(axis=1)
    lower, upper = percentile_interval(means, level=level)
    return Interval(
        point=float(np.mean(sample)), lower=lower, upper=upper, level=level, resamples=resamples
    )


def bootstrap_statistic_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    resamples: int = CLINICAL_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        raise ComparisonError("cannot bootstrap an empty sample")
    generator = rng_from(seed)
    draws = sample[bootstrap_indices(sample.size, resamples, generator)]
    statistics = np.asarray([statistic(draw) for draw in draws], dtype=float)
    lower, upper = percentile_interval(statistics, level=level)
    return Interval(
        point=float(statistic(sample)), lower=lower, upper=upper, level=level, resamples=resamples
    )


def paired_bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    resamples: int = CLINICAL_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap interval for the mean of ``first - second`` on paired observations."""

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ComparisonError("paired bootstrap needs equally sized samples")
    return bootstrap_mean_interval(a - b, resamples=resamples, level=level, seed=seed)


def cluster_bootstrap_difference(
    per_cluster_first: np.ndarray,
    per_cluster_second: np.ndarray,
    *,
    resamples: int = CLINICAL_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Reader-level cluster bootstrap of the mean paired difference."""

    a = np.asarray(per_cluster_first, dtype=float)
    b = np.asarray(per_cluster_second, dtype=float)
    if a.shape != b.shape:
        raise ComparisonError("cluster bootstrap needs one value per reader in each arm")
    if a.size == 0:
        raise ComparisonError("cluster bootstrap needs at least one reader")
    generator = rng_from(seed)
    draws = bootstrap_indices(a.size, resamples, generator)
    differences = (a - b)[draws].mean(axis=1)
    lower, upper = percentile_interval(differences, level=level)
    return Interval(
        point=float(np.mean(a - b)), lower=lower, upper=upper, level=level, resamples=resamples
    )


def cohens_kappa(first: np.ndarray, second: np.ndarray) -> float:
    """Cohen's kappa for two binary raters on the same cases."""

    a = np.asarray(first, dtype=int)
    b = np.asarray(second, dtype=int)
    if a.shape != b.shape:
        raise ComparisonError("kappa needs equally sized rating vectors")
    if a.size == 0:
        raise ComparisonError("kappa needs at least one case")
    observed = float(np.mean(a == b))
    pa_one = float(np.mean(a))
    pb_one = float(np.mean(b))
    expected = pa_one * pb_one + (1.0 - pa_one) * (1.0 - pb_one)
    if abs(1.0 - expected) <= 1e-12:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def paired_gap_summary(
    first: np.ndarray, second: np.ndarray, *, resamples: int = CLINICAL_RESAMPLES, seed: int = 0
) -> dict[str, float]:
    interval = paired_bootstrap_difference(first, second, resamples=resamples, seed=seed)
    return {
        "mean_difference": interval.point,
        "ci_lower": interval.lower,
        "ci_upper": interval.upper,
    }


def seed_level_summary(
    values: np.ndarray, *, resamples: int = SEED_RESAMPLES, seed: int = 0
) -> dict[str, float]:
    interval = bootstrap_mean_interval(values, resamples=resamples, seed=seed)
    sample = np.asarray(values, dtype=float)
    return {
        "mean": interval.point,
        "std": float(np.std(sample, ddof=1)) if sample.size > 1 else 0.0,
        "ci_lower": interval.lower,
        "ci_upper": interval.upper,
        "seeds": int(sample.size),
    }


def cochran_q(effects: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    """Cochran's Q and the I-squared statistic across sites."""

    theta = np.asarray(effects, dtype=float)
    variance = np.asarray(variances, dtype=float)
    if theta.shape != variance.shape:
        raise ComparisonError("effects and variances must align")
    if theta.size < 2:
        raise ComparisonError("Cochran's Q needs at least two strata")
    if np.any(variance <= 0.0):
        raise ComparisonError("every stratum variance must be positive")
    weights = 1.0 / variance
    pooled = float(np.sum(weights * theta) / np.sum(weights))
    q = float(np.sum(weights * (theta - pooled) ** 2))
    degrees = theta.size - 1
    heterogeneity = max(0.0, (q - degrees) / q) if q > 0 else 0.0
    return {
        "pooled": pooled,
        "q": q,
        "degrees_of_freedom": float(degrees),
        "i_squared": heterogeneity,
        "flag_heterogeneity": 1.0 if heterogeneity > 0.5 else 0.0,
        "max_gap": float(np.max(theta) - np.min(theta)),
    }
