"""Flux sampling inside the feasible polytope at a fixed growth floor.

Ref: Sec. 4.5 family (4) and the FluxGAT-style comparator, which samples fluxes and
feeds them to a graph model; the sampling used here is deterministic per seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

from crb_concordance.metabolism import fba
from crb_concordance.metabolism.model import StoichiometricModel
from crb_concordance.utils.numerics import rng_from

WARMUP_STEPS = 50
STEP_TOLERANCE = 1e-9


class SamplingError(ValueError):
    """Raised when a flux sample cannot be drawn from the feasible set."""


@dataclass(frozen=True, slots=True)
class FluxSampleSet:
    reaction_ids: tuple[str, ...]
    samples: np.ndarray
    growth_floor: float
    seed: int

    def column(self, reaction_id: str) -> np.ndarray:
        try:
            position = self.reaction_ids.index(reaction_id)
        except ValueError as error:
            raise SamplingError(f"reaction not sampled: {reaction_id}") from error
        return self.samples[:, position]

    def mean_flux(self, reaction_id: str) -> float:
        return float(np.mean(self.column(reaction_id)))

    def flux_shift(self, other: FluxSampleSet, pathway_reactions: tuple[str, ...]) -> float:
        """Mean absolute flux change over a pathway between two sample sets."""

        present = [name for name in pathway_reactions if name in self.reaction_ids]
        if not present:
            return 0.0
        return float(
            np.mean([abs(self.mean_flux(name) - other.mean_flux(name)) for name in present])
        )


def _step_bounds(
    point: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    growth_direction: float,
    slack: float,
) -> tuple[float, float]:
    """Exact feasible step interval along a direction that preserves mass balance."""

    low = -np.inf
    high = np.inf
    for index in range(point.size):
        component = direction[index]
        if component > STEP_TOLERANCE:
            high = min(high, (upper[index] - point[index]) / component)
            low = max(low, (lower[index] - point[index]) / component)
        elif component < -STEP_TOLERANCE:
            high = min(high, (lower[index] - point[index]) / component)
            low = max(low, (upper[index] - point[index]) / component)
    # g(point + t d) = g(point) + t (g . d) must stay above the floor, so the growth
    # term bounds t on the side opposite to the sign of (g . d).
    if growth_direction > STEP_TOLERANCE:
        low = max(low, -slack / growth_direction)
    elif growth_direction < -STEP_TOLERANCE:
        high = min(high, -slack / growth_direction)
    return low, high


def sample_fluxes(
    model: StoichiometricModel,
    *,
    n_samples: int,
    seed: int,
    optimum_fraction: float = 0.9,
) -> FluxSampleSet:
    """Hit-and-run sampling with the growth optimum held above a fixed fraction.

    Directions are drawn in the null space of the stoichiometric matrix, so every step
    preserves mass balance exactly and the feasible interval comes from the box and
    growth-floor constraints alone.
    """

    if n_samples < 1:
        raise SamplingError(f"n_samples must be positive: {n_samples!r}")
    if not 0.0 < optimum_fraction <= 1.0:
        raise SamplingError(f"optimum_fraction must lie in (0, 1]: {optimum_fraction!r}")
    optimum = fba.max_biomass(model)
    if not optimum.optimal:
        raise SamplingError("baseline growth problem is infeasible")
    floor = float(optimum_fraction * optimum.objective_value)
    growth_row = model.objective_vector()
    # Reactions held at a single value (a knock-out, or an upper bound of zero) cannot
    # move, so directions are drawn over the free reactions only.
    free = np.flatnonzero(model.upper - model.lower > STEP_TOLERANCE)
    if free.size == 0:
        raise SamplingError("no reaction of the model can carry a varying flux")
    null_basis = null_space(model.stoichiometry.toarray()[:, free])
    if null_basis.shape[1] == 0:
        raise SamplingError("the free reactions admit no mass-balancing redistribution")
    generator = rng_from(seed)
    point = np.asarray(optimum.fluxes, dtype=float).copy()
    point = np.clip(point, model.lower, model.upper)
    drawn = np.zeros((n_samples, model.n_reactions))
    accepted = 0
    attempts = 0
    budget = (WARMUP_STEPS + n_samples) * 20
    while accepted < n_samples and attempts < budget:
        attempts += 1
        coefficients = generator.normal(size=null_basis.shape[1])
        direction = np.zeros(model.n_reactions)
        direction[free] = null_basis @ coefficients
        norm = float(np.linalg.norm(direction))
        if norm <= STEP_TOLERANCE:
            continue
        direction = direction / norm
        slack = float(growth_row @ point) - floor
        growth_direction = float(growth_row @ direction)
        low, high = _step_bounds(
            point, direction, model.lower, model.upper, growth_direction, slack
        )
        if not np.isfinite(low) or not np.isfinite(high) or high - low <= STEP_TOLERANCE:
            continue
        step = generator.uniform(low, high)
        candidate = point + step * direction
        if not model.is_feasible(candidate, tolerance=1e-6):
            continue
        if float(growth_row @ candidate) < floor - 1e-6:
            continue
        point = candidate
        if attempts > WARMUP_STEPS:
            drawn[accepted] = point
            accepted += 1
    if accepted < n_samples:
        raise SamplingError(f"drew only {accepted} of {n_samples} requested samples")
    return FluxSampleSet(
        reaction_ids=tuple(entry.identifier for entry in model.reactions),
        samples=drawn,
        growth_floor=floor,
        seed=int(seed),
    )


def flux_score_distribution(samples: FluxSampleSet, reaction_id: str) -> dict[str, float]:
    column = samples.column(reaction_id)
    return {
        "mean": float(np.mean(column)),
        "std": float(np.std(column)),
        "minimum": float(np.min(column)),
        "maximum": float(np.max(column)),
    }


def range_narrowing(wide: FluxSampleSet, narrow: FluxSampleSet, reaction_id: str) -> float:
    """Fraction by which a knock-out narrows the achievable range of a reaction."""

    wide_column = wide.column(reaction_id)
    narrow_column = narrow.column(reaction_id)
    wide_span = float(np.max(wide_column) - np.min(wide_column))
    if wide_span <= 1e-12:
        return 0.0
    narrow_span = float(np.max(narrow_column) - np.min(narrow_column))
    return max(0.0, 1.0 - narrow_span / wide_span)
