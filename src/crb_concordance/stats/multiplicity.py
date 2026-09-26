"""Multiplicity control for the pre-specified statistical families.

Ref: Sec. 4.6 (multiplicity across the pre-specified primary family is controlled by
Holm-Bonferroni and across the larger exploratory subgroup and endpoint battery by
Benjamini-Hochberg, with both families and corrections declared in advance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MultiplicityError(ValueError):
    """Raised when a multiplicity request is malformed."""


@dataclass(frozen=True, slots=True)
class AdjustedResult:
    """One hypothesis with its raw and adjusted p-value."""

    name: str
    p_value: float
    adjusted: float
    rejected: bool

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "name": self.name,
            "p_value": self.p_value,
            "adjusted": self.adjusted,
            "rejected": self.rejected,
        }


def _validate(p_values: list[float], alpha: float) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        raise MultiplicityError("no p-values supplied")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise MultiplicityError("p-values must lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise MultiplicityError(f"alpha must lie in (0, 1): {alpha!r}")
    return values


def holm_bonferroni(
    names: tuple[str, ...], p_values: list[float], *, alpha: float = 0.05
) -> tuple[AdjustedResult, ...]:
    """Step-down Holm-Bonferroni control of the family-wise error rate."""

    values = _validate(p_values, alpha)
    if len(names) != values.size:
        raise MultiplicityError("one name is required per p-value")
    order = np.argsort(values, kind="stable")
    total = values.size
    adjusted = np.empty(total, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        scaled = min(1.0, (total - rank) * float(values[index]))
        running = max(running, scaled)
        adjusted[index] = running
    return tuple(
        AdjustedResult(
            name=names[index],
            p_value=float(values[index]),
            adjusted=float(adjusted[index]),
            rejected=bool(adjusted[index] <= alpha),
        )
        for index in range(total)
    )


def benjamini_hochberg(
    names: tuple[str, ...], p_values: list[float], *, alpha: float = 0.05
) -> tuple[AdjustedResult, ...]:
    """Step-up Benjamini-Hochberg control of the false discovery rate."""

    values = _validate(p_values, alpha)
    if len(names) != values.size:
        raise MultiplicityError("one name is required per p-value")
    order = np.argsort(-values, kind="stable")
    total = values.size
    adjusted = np.empty(total, dtype=float)
    running = 1.0
    for position, index in enumerate(order):
        rank = total - position
        scaled = min(1.0, float(values[index]) * total / rank)
        running = min(running, scaled)
        adjusted[index] = running
    return tuple(
        AdjustedResult(
            name=names[index],
            p_value=float(values[index]),
            adjusted=float(adjusted[index]),
            rejected=bool(adjusted[index] <= alpha),
        )
        for index in range(total)
    )


def holm_rejects_only_uncorrected_rejections(
    names: tuple[str, ...], p_values: list[float], *, alpha: float = 0.05
) -> bool:
    """Holm never rejects a hypothesis whose uncorrected p-value exceeds alpha.

    The adjusted values are at least the raw ones by construction, so the step-down
    correction cannot admit a hypothesis that unadjusted testing would refuse.
    """

    holm = holm_bonferroni(names, p_values, alpha=alpha)
    return all(
        (not holm[index].rejected) or p_values[index] <= alpha for index in range(len(p_values))
    )


def adjusted_never_below_raw(
    names: tuple[str, ...], p_values: list[float], *, alpha: float = 0.05
) -> bool:
    """Every adjusted p-value is at least its raw counterpart."""

    holm = holm_bonferroni(names, p_values, alpha=alpha)
    return all(holm[index].adjusted >= p_values[index] - 1e-12 for index in range(len(p_values)))


def family_plan() -> dict[str, dict[str, object]]:
    """The two declared families and their corrections."""

    return {
        "primary": {
            "correction": "Holm-Bonferroni",
            "controls": "family-wise error rate",
            "members": [
                "pCR-discrimination AUROC against the standard-of-care nomogram",
                "reader-study assisted-minus-unaided kappa gain",
                "cross-site consistency of the primary AUROC",
            ],
        },
        "exploratory": {
            "correction": "Benjamini-Hochberg",
            "controls": "false discovery rate",
            "members": [
                "subgroup and endpoint battery declared in advance",
                "site-level and region-level gaps",
                "interval-width association with cross-modal conflict",
            ],
        },
    }
