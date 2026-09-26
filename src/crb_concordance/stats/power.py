"""Statistics of the pre-specified clinical analysis plan.

Ref: Sec. 4.6 (Hanley-McNeil sizing for two correlated AUROC curves on the same
cohort; DeLong's method for the difference between two correlated ROC curves;
bootstrap confidence intervals; Cochran's Q with I-squared; Holm-Bonferroni and
Benjamini-Hochberg multiplicity control; stratified Cox models with competing-risk
handling; the reader-study mixed-effects logistic model with a cluster bootstrap).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf

import numpy as np

from crb_concordance.utils.numerics import clip01, normal_ppf

TWO_SIDED_ALPHA = 0.05
POWER = 0.80


class PowerError(ValueError):
    """Raised when a sizing request is inconsistent."""

    pass


@dataclass(frozen=True, slots=True)
class SizingInputs:
    """The declared inputs of the prospective-arm sizing calculation."""

    baseline_auroc: float = 0.70
    margin: float = 0.081
    prevalence: float = 0.25
    alpha: float = TWO_SIDED_ALPHA
    power: float = POWER

    def validate(self) -> None:
        if not 0.5 < self.baseline_auroc < 1.0:
            raise PowerError(f"baseline AUROC must lie in (0.5, 1): {self.baseline_auroc!r}")
        if not 0.0 < self.margin < 1.0 - self.baseline_auroc:
            raise PowerError(f"margin outside the reachable range: {self.margin!r}")
        if not 0.0 < self.prevalence < 1.0:
            raise PowerError(f"prevalence must lie in (0, 1): {self.prevalence!r}")
        if not 0.0 < self.alpha < 0.5:
            raise PowerError(f"alpha must lie in (0, 0.5): {self.alpha!r}")
        if not 0.0 < self.power < 1.0:
            raise PowerError(f"power must lie in (0, 1): {self.power!r}")

    @property
    def combined_auroc(self) -> float:
        return self.baseline_auroc + self.margin

    def as_dict(self) -> dict[str, float]:
        return {
            "baseline_auroc": self.baseline_auroc,
            "combined_auroc": self.combined_auroc,
            "margin": self.margin,
            "prevalence": self.prevalence,
            "alpha": self.alpha,
            "power": self.power,
        }


def hanley_mcneil_variance(auroc: float, positives: float, negatives: float) -> float:
    """Sampling variance of one AUROC estimator under the Hanley-McNeil formula."""

    if positives <= 1.0 or negatives <= 1.0:
        raise PowerError("the Hanley-McNeil formula needs at least two cases per class")
    q1 = auroc / (2.0 - auroc)
    q2 = 2.0 * auroc * auroc / (1.0 + auroc)
    numerator = (
        auroc * (1.0 - auroc)
        + (positives - 1.0) * (q1 - auroc * auroc)
        + (negatives - 1.0) * (q2 - auroc * auroc)
    )
    return numerator / (positives * negatives)


def paired_variance(
    baseline: float, combined: float, positives: float, negatives: float, correlation: float
) -> float:
    """Variance of the paired AUROC difference at the assumed correlation."""

    first = hanley_mcneil_variance(baseline, positives, negatives)
    second = hanley_mcneil_variance(combined, positives, negatives)
    return first + second - 2.0 * correlation * float(np.sqrt(first * second))


def power_at_size(inputs: SizingInputs, records: int, correlation: float) -> float:
    """Power of the two-sided paired AUROC comparison at a given accrual."""

    inputs.validate()
    if records <= 0:
        raise PowerError(f"accrual must be positive: {records!r}")
    positives = inputs.prevalence * records
    negatives = records - positives
    variance = paired_variance(
        inputs.baseline_auroc, inputs.combined_auroc, positives, negatives, correlation
    )
    if variance <= 0.0:
        raise PowerError("paired variance is not positive at this correlation")
    critical = normal_ppf(1.0 - inputs.alpha / 2.0)
    shift = inputs.margin / float(np.sqrt(variance))
    return float(clip01(normal_cdf(shift - critical) + normal_cdf(-shift - critical)))


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / float(np.sqrt(2.0))))


def analytic_minimum_records(
    inputs: SizingInputs, correlation: float, *, ceiling: int = 20000
) -> int:
    """Smallest accrual whose power reaches the declared level."""

    inputs.validate()
    z_alpha = normal_ppf(1.0 - inputs.alpha / 2.0)
    z_beta = normal_ppf(inputs.power)
    for records in range(20, ceiling + 1, 2):
        positives = inputs.prevalence * records
        negatives = records - positives
        variance = paired_variance(
            inputs.baseline_auroc, inputs.combined_auroc, positives, negatives, correlation
        )
        if variance <= 0.0:
            continue
        if inputs.margin / float(np.sqrt(variance)) >= z_alpha + z_beta:
            return records
    raise PowerError(f"no accrual below {ceiling} reaches the declared power")


@dataclass(frozen=True, slots=True)
class SizingResult:
    """The sizing table at every declared correlation assumption."""

    inputs: SizingInputs
    correlations: tuple[float, ...]
    minimum_records: tuple[int, ...]
    pre_specified_targets: tuple[int, ...]

    def as_rows(self) -> list[dict[str, float]]:
        return [
            {
                "correlation": correlation,
                "analytic_minimum_records": float(minimum),
                "pre_specified_target_records": float(target),
                "target_covers_minimum": 1.0 if target >= minimum else 0.0,
                "power_at_target": power_at_size(self.inputs, target, correlation),
            }
            for correlation, minimum, target in zip(
                self.correlations, self.minimum_records, self.pre_specified_targets, strict=True
            )
        ]

    def as_dict(self) -> dict[str, object]:
        return {"inputs": self.inputs.as_dict(), "rows": self.as_rows()}


DECLARED_CORRELATIONS: tuple[float, ...] = (0.3, 0.4, 0.5)
DECLARED_TARGETS: tuple[int, ...] = (690, 600, 500)
ACCRUAL_RANGE: tuple[int, int] = (750, 900)


def prospective_sizing(
    inputs: SizingInputs | None = None,
    *,
    correlations: tuple[float, ...] = DECLARED_CORRELATIONS,
    targets: tuple[int, ...] = DECLARED_TARGETS,
) -> SizingResult:
    settings = inputs or SizingInputs()
    settings.validate()
    if len(correlations) != len(targets):
        raise PowerError("every correlation assumption needs a declared target")
    minimum = tuple(analytic_minimum_records(settings, correlation) for correlation in correlations)
    return SizingResult(
        inputs=settings,
        correlations=correlations,
        minimum_records=minimum,
        pre_specified_targets=targets,
    )


def accrual_range_covers(result: SizingResult) -> dict[str, object]:
    low, high = ACCRUAL_RANGE
    rows = result.as_rows()
    covered = [row for row in rows if row["analytic_minimum_records"] <= low]
    return {
        "accrual_range": [low, high],
        "assumptions_covered_by_low_end": len(covered),
        "assumptions_total": len(rows),
        "all_covered_at_low_end": len(covered) == len(rows),
    }
