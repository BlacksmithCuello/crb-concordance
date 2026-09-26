"""Flux-balance, variability and knockout analysis on the stoichiometric model.

Ref: Sec. 4.2 item (3) (feasibility of alternative pathways under suppression of g);
Sec. 4.5 family (4) (classical unassisted flux-balance analysis is the comparator).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from crb_concordance.metabolism.catalogue import GLYCOLYSIS, OXPHOS, pathway_reactions
from crb_concordance.metabolism.model import StoichiometricModel

INFEASIBLE = 2
OPTIMAL = 0
FVA_CAP = 1e6


class FluxError(ValueError):
    """Raised when a flux problem is malformed."""


@dataclass(frozen=True, slots=True)
class FBAResult:
    status: str
    objective_value: float
    fluxes: np.ndarray
    residual: float

    @property
    def optimal(self) -> bool:
        return self.status == "optimal"


@dataclass(frozen=True, slots=True)
class FluxRange:
    reaction_id: str
    minimum: float
    maximum: float

    @property
    def span(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True, slots=True)
class VariabilityResult:
    ranges: tuple[FluxRange, ...]
    fraction: float
    optimum: float

    def as_dict(self) -> dict[str, float]:
        return {entry.reaction_id: entry.maximum for entry in self.ranges}

    def span_of(self, reaction_id: str) -> float:
        for entry in self.ranges:
            if entry.reaction_id == reaction_id:
                return entry.span
        raise FluxError(f"reaction not covered by this variability scan: {reaction_id}")


@dataclass(frozen=True, slots=True)
class KnockoutGrowth:
    gene: str
    baseline: float
    suppressed: float
    blocked_reactions: tuple[str, ...]

    @property
    def growth_ratio(self) -> float:
        if self.baseline <= 1e-12:
            return 0.0
        return self.suppressed / self.baseline

    @property
    def loss_of_fitness(self) -> float:
        return 1.0 - self.growth_ratio

    def as_dict(self) -> dict[str, float | str | list[str]]:
        return {
            "gene": self.gene,
            "baseline": self.baseline,
            "suppressed": self.suppressed,
            "growth_ratio": self.growth_ratio,
            "loss_of_fitness": self.loss_of_fitness,
            "blocked_reactions": list(self.blocked_reactions),
        }


def _bounds(
    model: StoichiometricModel, extra: list[tuple[float, float]] | None
) -> list[tuple[float, float]]:
    pairs = list(zip(model.lower.tolist(), model.upper.tolist(), strict=True))
    for lower, upper in extra or []:
        if lower > upper:
            raise FluxError(f"inconsistent bound pair: ({lower}, {upper})")
        pairs.append((lower, upper))
    return pairs


def solve(
    model: StoichiometricModel,
    *,
    objective: np.ndarray | None = None,
    maximise: bool = True,
    extra_equalities: list[tuple[np.ndarray, float]] | None = None,
    extra_bounds: list[tuple[float, float]] | None = None,
) -> FBAResult:
    """Solve one linear program over the steady-state flux polytope."""

    vector = model.objective_vector() if objective is None else np.asarray(objective, dtype=float)
    if vector.shape != (model.n_reactions,):
        raise FluxError("objective vector shape does not match the reaction count")
    rows = [model.stoichiometry]
    target = [np.zeros(model.n_metabolites)]
    for row, value in extra_equalities or []:
        rows.append(np.asarray(row, dtype=float).reshape(1, -1))
        target.append(np.asarray([value], dtype=float))
    equality = np.vstack([np.asarray(r.todense()) if hasattr(r, "todense") else r for r in rows])
    right = np.concatenate(target)
    coefficients = -vector if maximise else vector
    outcome = linprog(
        coefficients,
        A_eq=equality,
        b_eq=right,
        bounds=_bounds(model, extra_bounds),
        method="highs",
    )
    if not outcome.success or outcome.x is None:
        return FBAResult(
            status="infeasible",
            objective_value=float("nan"),
            fluxes=np.zeros(model.n_reactions),
            residual=float("nan"),
        )
    fluxes = np.asarray(outcome.x, dtype=float)
    return FBAResult(
        status="optimal",
        objective_value=float(vector @ fluxes),
        fluxes=fluxes,
        residual=model.mass_balance_residual(fluxes),
    )


def max_biomass(model: StoichiometricModel) -> FBAResult:
    return solve(model, maximise=True)


def min_biomass(model: StoichiometricModel) -> FBAResult:
    return solve(model, maximise=False)


def _extreme_flux(
    model: StoichiometricModel,
    reaction_id: str,
    *,
    floor: float | None,
    maximise: bool,
    objective_id: str | None = None,
) -> float:
    vector = np.zeros(model.n_reactions)
    vector[model.index_of(reaction_id)] = 1.0
    extra_equalities: list[tuple[np.ndarray, float]] | None = None
    if floor is not None:
        growth = np.zeros(model.n_reactions)
        growth[model.index_of(objective_id or model.objective_id)] = 1.0
        extra_equalities = [(growth, float(floor))]
    result = solve(model, objective=vector, maximise=maximise, extra_equalities=extra_equalities)
    if not result.optimal:
        return 0.0
    return float(result.fluxes[model.index_of(reaction_id)])


def maximise_reaction(
    model: StoichiometricModel,
    reaction_id: str,
    *,
    floor: float | None = None,
    objective_id: str | None = None,
) -> float:
    """Largest achievable flux through one reaction, optionally at a fixed growth floor."""

    return _extreme_flux(model, reaction_id, floor=floor, maximise=True, objective_id=objective_id)


def minimise_reaction(
    model: StoichiometricModel,
    reaction_id: str,
    *,
    floor: float | None = None,
    objective_id: str | None = None,
) -> float:
    return _extreme_flux(model, reaction_id, floor=floor, maximise=False, objective_id=objective_id)


def flux_variability(
    model: StoichiometricModel,
    *,
    fraction: float = 1.0,
    reactions: tuple[str, ...] | None = None,
    objective_id: str | None = None,
) -> VariabilityResult:
    """Range of every reaction compatible with at least ``fraction`` of the optimum."""

    if not 0.0 < fraction <= 1.0:
        raise FluxError(f"fraction must lie in (0, 1]: {fraction!r}")
    optimum = max_biomass(model)
    if not optimum.optimal:
        raise FluxError("baseline growth problem is infeasible")
    floor = float(fraction * optimum.objective_value)
    selected = reactions or tuple(entry.identifier for entry in model.reactions)
    ranges: list[FluxRange] = []
    for reaction_id in selected:
        ranges.append(
            FluxRange(
                reaction_id=reaction_id,
                minimum=minimise_reaction(
                    model, reaction_id, floor=floor, objective_id=objective_id
                ),
                maximum=maximise_reaction(
                    model, reaction_id, floor=floor, objective_id=objective_id
                ),
            )
        )
    return VariabilityResult(
        ranges=tuple(ranges), fraction=fraction, optimum=float(optimum.objective_value)
    )


def knockout_growth(model: StoichiometricModel, gene: str) -> KnockoutGrowth:
    """Compare growth before and after invalidating every reaction the gene gates."""

    baseline = max_biomass(model)
    if not baseline.optimal:
        raise FluxError("baseline growth problem is infeasible")
    suppressed_model = model.knock_out(gene)
    suppressed = max_biomass(suppressed_model)
    value = float(suppressed.objective_value) if suppressed.optimal else 0.0
    return KnockoutGrowth(
        gene=gene,
        baseline=float(baseline.objective_value),
        suppressed=value,
        blocked_reactions=model.blocked_reactions(gene),
    )


def pathway_utilisation(
    model: StoichiometricModel, pathway: str, *, fraction: float = 0.9
) -> float:
    """Total flux carried by a pathway at a fixed fraction of the growth optimum."""

    entries = pathway_reactions(model.reactions, pathway)
    if not entries:
        return 0.0
    optimum = max_biomass(model)
    if not optimum.optimal:
        return 0.0
    floor = fraction * float(optimum.objective_value)
    total = 0.0
    for entry in entries:
        total += abs(maximise_reaction(model, entry.identifier, floor=floor))
    return total


def energy_coupling_ratio(model: StoichiometricModel) -> float:
    """Lactate export flux per unit glycolytic uptake, the Warburg-style ratio."""

    result = max_biomass(model)
    if not result.optimal:
        return 0.0
    uptake = abs(float(result.fluxes[model.index_of("EX_glc_D_e")]))
    if uptake <= 1e-12:
        return 0.0
    exported = float(result.fluxes[model.index_of("MCT1_export")]) + float(
        result.fluxes[model.index_of("MCT4_export")]
    )
    return exported / uptake


def pathway_flux_snapshot(model: StoichiometricModel) -> dict[str, float]:
    result = max_biomass(model)
    if not result.optimal:
        return {}
    snapshot: dict[str, float] = {
        "biomass": float(result.fluxes[model.index_of(model.objective_id)])
    }
    for pathway in (GLYCOLYSIS, OXPHOS):
        snapshot[pathway] = sum(
            abs(float(result.fluxes[model.index_of(entry.identifier)]))
            for entry in pathway_reactions(model.reactions, pathway)
        )
    return snapshot
