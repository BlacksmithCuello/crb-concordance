"""The two design principles stated as Propositions 1 and 2.

Ref: Sec. 4.1, Proposition 1 (conflict monotonicity) and Proposition 2
(pignistic convergence under independence, up to a fixed-discount floor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.belief.combine import CombinationResult, combine_sequential
from crb_concordance.belief.discount import DiscountRates, discount
from crb_concordance.belief.mass import MassTriple, from_posterior
from crb_concordance.utils.numerics import clip01, sigmoid
from crb_concordance.utils.types import Modality


@dataclass(frozen=True, slots=True)
class GuaranteeCheck:
    name: str
    holds: bool
    detail: str


@dataclass(frozen=True, slots=True)
class MonotonicityScan:
    """Interval width and accumulated conflict along a pre-aggregation-fixed path."""

    pressure: np.ndarray
    total_conflict: np.ndarray
    width: np.ndarray
    pignistic: np.ndarray

    def monotone_non_decreasing(self, tolerance: float = 1e-12) -> bool:
        return bool(np.all(np.diff(self.width) >= -tolerance))

    def worst_decrease(self) -> float:
        deltas = np.diff(self.width)
        return float(-min(0.0, float(np.min(deltas)))) if deltas.size else 0.0


@dataclass(frozen=True, slots=True)
class ConvergenceScan:
    """Bias of the pignistic point estimate as within-modality evidence sharpens."""

    sharpness: np.ndarray
    pignistic: np.ndarray
    bias_to_truth: np.ndarray
    limit: float
    floor: float

    def approaches_limit(self, tolerance: float = 1e-6) -> bool:
        return bool(abs(float(self.pignistic[-1]) - self.limit) <= tolerance)

    def converges_to_floor(self, tolerance: float = 1e-6) -> bool:
        return bool(abs(float(self.bias_to_truth[-1]) - self.floor) <= tolerance)


def interval_width_trajectory(triples: list[MassTriple]) -> tuple[float, ...]:
    """Widths after every sequential step, starting from the first mass."""

    widths = [triples[0].interval_width()]
    current = triples[0]
    for triple in triples[1:]:
        result = combine_sequential([current, triple])
        current = result.triple
        widths.append(current.interval_width())
    return tuple(widths)


def conflict_monotonicity_scan(
    rates: DiscountRates,
    *,
    pressure_grid: np.ndarray | None = None,
    center: float = 0.5,
    spread: float = 0.45,
) -> MonotonicityScan:
    """Vary pairwise agreement while holding every pre-aggregation ignorance fixed.

    Reported modality k enters with m^alpha_k(Theta) = 1 - alpha_k whatever the
    pressure, so any width change is attributable to conflict alone, which is the
    setting Proposition 1 quantifies over.
    """

    grid = (
        np.linspace(0.0, 1.0, 26)
        if pressure_grid is None
        else np.asarray(pressure_grid, dtype=float)
    )
    order = rates.reliability_order()
    conflicts = np.zeros(grid.size)
    widths = np.zeros(grid.size)
    pignistics = np.zeros(grid.size)
    for position, pressure in enumerate(grid):
        triples: list[MassTriple] = []
        for index, modality in enumerate(order):
            alpha = rates.of(modality)
            sign = 1.0 if index % 2 == 0 else -1.0
            posterior = clip01(center + spread * float(pressure) * sign)
            triples.append(
                MassTriple(
                    v=alpha * posterior,
                    n=alpha * (1.0 - posterior),
                    theta=1.0 - alpha,
                )
            )
        result = combine_sequential(triples, [modality.value for modality in order])
        conflicts[position] = result.total_conflict
        widths[position] = result.interval_width
        pignistics[position] = result.pignistic
    return MonotonicityScan(
        pressure=grid, total_conflict=conflicts, width=widths, pignistic=pignistics
    )


def pignistic_bias_floor(rates: DiscountRates) -> float:
    """Half the product of the per-modality knowledge gaps (Proposition 2)."""

    return 0.5 * rates.ignorance_floor()


def zero_conflict_limit(rates: DiscountRates, *, target: bool = True) -> tuple[float, float]:
    """BetP(V) and its distance from the truth when every modality agrees exactly."""

    floor = rates.ignorance_floor()
    if target:
        limit = 1.0 - 0.5 * floor
    else:
        limit = 0.5 * floor
    return limit, 0.5 * floor


def floor_vanishes_iff_rate_saturates(rates: DiscountRates) -> GuaranteeCheck:
    floor = rates.ignorance_floor()
    saturates = any(rates.of(modality) >= 1.0 for modality in Modality)
    holds = (floor == 0.0) == saturates
    return GuaranteeCheck(
        name="floor_vanishes_iff_rate_saturates",
        holds=bool(holds),
        detail=f"floor={floor:.12g}, any_rate_one={saturates}",
    )


def conflict_monotonicity_check(scan: MonotonicityScan) -> GuaranteeCheck:
    holds = scan.monotone_non_decreasing()
    return GuaranteeCheck(
        name="conflict_monotonicity",
        holds=holds,
        detail=(
            f"width {scan.width[0]:.6f} -> {scan.width[-1]:.6f} over "
            f"conflict {scan.total_conflict[0]:.6f} -> {scan.total_conflict[-1]:.6f}; "
            f"worst decrease {scan.worst_decrease():.3e}"
        ),
    )


def pignistic_convergence_scan(
    rates: DiscountRates,
    *,
    sharpness_grid: np.ndarray | None = None,
    target: bool = True,
) -> ConvergenceScan:
    """Sharpen every modality towards the same status and track the residual bias."""

    grid = (
        np.linspace(0.25, 24.0, 48)
        if sharpness_grid is None
        else np.asarray(sharpness_grid, dtype=float)
    )
    order = rates.reliability_order()
    sign = 1.0 if target else -1.0
    betps = np.zeros(grid.size)
    for position, sharpness in enumerate(grid):
        triples: list[MassTriple] = []
        for index, modality in enumerate(order):
            strength = 0.6 + 0.4 * index
            posterior = float(sigmoid(sign * strength * float(sharpness)))
            alpha = rates.of(modality)
            triples.append(discount(from_posterior(posterior), alpha))
        result: CombinationResult = combine_sequential(
            triples, [modality.value for modality in order]
        )
        betps[position] = result.pignistic
    limit, floor = zero_conflict_limit(rates, target=target)
    truth = 1.0 if target else 0.0
    return ConvergenceScan(
        sharpness=grid,
        pignistic=betps,
        bias_to_truth=np.abs(betps - truth),
        limit=limit,
        floor=floor,
    )


def proposition_summary(rates: DiscountRates) -> dict[str, GuaranteeCheck]:
    monotonicity = conflict_monotonicity_check(conflict_monotonicity_scan(rates))
    convergence = pignistic_convergence_scan(rates)
    floor_scan = pignistic_convergence_scan(rates, target=False)
    convergence_holds = convergence.approaches_limit() and convergence.converges_to_floor()
    return {
        "proposition_1": monotonicity,
        "proposition_2": GuaranteeCheck(
            name="pignistic_convergence",
            holds=bool(convergence_holds and floor_scan.approaches_limit()),
            detail=(
                f"limit {convergence.limit:.8f} reached {convergence.pignistic[-1]:.8f}; "
                f"bias {convergence.bias_to_truth[-1]:.8f} against floor {convergence.floor:.8f}"
            ),
        ),
        "floor_vanishes": floor_vanishes_iff_rate_saturates(rates),
    }
