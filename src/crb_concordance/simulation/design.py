"""The in silico simulation design of Supplementary Table S1.

Ref: Sec. 2 (the in silico simulation implements the distribution of conflict based
on previously defined synthetic evidence parameters and is the one study carried
out); Supplementary Table S1 (simulation seeds 1-20, 2,000 simulated candidates per
seed, 15% simulated ground-truth prevalence, conflict-probability grid 0.1/0.3/0.5,
absent-evidence-fraction grid 0/0.25/0.5, 5,000 seed-level bootstrap resamples).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crb_concordance.cohorts.generator import (
    EvidenceDesign,
    SimulatedCandidate,
    simulate_candidates,
)
from crb_concordance.utils.types import Modality

SIMULATION_SEEDS: tuple[int, ...] = tuple(range(1, 21))
CONFLICT_GRID: tuple[float, ...] = (0.1, 0.3, 0.5)
ABSENT_GRID: tuple[float, ...] = (0.0, 0.25, 0.5)
SIMULATED_CANDIDATES = 2000
SIMULATED_PREVALENCE = 0.15
SEED_BOOTSTRAP_RESAMPLES = 5000
PRIMARY_RECALL_CUTOFF = 10
CONCORDANCE_DISAGREEMENT_MARGIN = 0.25
PARITY_MARGIN = 0.15


class SimulationError(ValueError):
    """Raised when a simulation cell cannot be assembled."""


@dataclass(frozen=True, slots=True)
class SimulationCell:
    """One point of the (conflict, absent) grid evaluated over every seed."""

    conflict_probability: float
    absent_fraction: float
    seeds: tuple[int, ...] = SIMULATION_SEEDS
    n_candidates: int = SIMULATED_CANDIDATES
    prevalence: float = SIMULATED_PREVALENCE

    @property
    def label(self) -> str:
        return f"conflict={self.conflict_probability:g};absent={self.absent_fraction:g}"

    def evidence_design(self, seed: int) -> EvidenceDesign:
        design = EvidenceDesign(
            n_candidates=self.n_candidates,
            prevalence=self.prevalence,
            conflict_probability=self.conflict_probability,
            absent_fraction=self.absent_fraction,
            seed=seed,
        )
        design.validate()
        return design

    def candidates(self, seed: int) -> tuple[SimulatedCandidate, ...]:
        return simulate_candidates(self.evidence_design(seed))

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "conflict_probability": self.conflict_probability,
            "absent_fraction": self.absent_fraction,
            "seeds": list(self.seeds),
            "n_candidates": self.n_candidates,
            "prevalence": self.prevalence,
        }


@dataclass(frozen=True, slots=True)
class SimulationDesign:
    """The full grid: every (conflict, absent) combination over every seed."""

    cells: tuple[SimulationCell, ...] = field(
        default_factory=lambda: tuple(
            SimulationCell(conflict_probability=conflict, absent_fraction=absent)
            for conflict in CONFLICT_GRID
            for absent in ABSENT_GRID
        )
    )
    bootstrap_resamples: int = SEED_BOOTSTRAP_RESAMPLES
    recall_cutoff: int = PRIMARY_RECALL_CUTOFF
    concordance_margin: float = CONCORDANCE_DISAGREEMENT_MARGIN
    parity_margin: float = PARITY_MARGIN

    def validate(self) -> None:
        if not self.cells:
            raise SimulationError("the simulation grid is empty")
        if self.bootstrap_resamples < 1:
            raise SimulationError("bootstrap resamples must be positive")
        if self.recall_cutoff < 1:
            raise SimulationError("the recall cutoff must be positive")

    @property
    def total_runs(self) -> int:
        return sum(len(cell.seeds) for cell in self.cells)

    @property
    def saturated_cutoff(self) -> int:
        """A cutoff at which the pre-specified margin is reachable at all.

        With a pool of N candidates and prevalence p the top-k recall is bounded by
        k/(p N), so the declared 15-point margin is unreachable whenever
        k < 0.15 p N. The saturated cutoff is the first integer at which the margin
        can in principle be met, and the inequality is reported at both cutoffs.
        """

        first = self.cells[0] if self.cells else None
        if first is None:
            raise SimulationError("the simulation grid is empty")
        bound = self.parity_margin * first.prevalence * first.n_candidates
        return max(self.recall_cutoff, int(np.ceil(bound)))

    def as_dict(self) -> dict[str, object]:
        return {
            "cells": [cell.as_dict() for cell in self.cells],
            "bootstrap_resamples": self.bootstrap_resamples,
            "recall_cutoff": self.recall_cutoff,
            "concordance_margin": self.concordance_margin,
            "saturated_cutoff": self.saturated_cutoff,
            "parity_margin": self.parity_margin,
            "total_runs": self.total_runs,
        }


@dataclass(frozen=True, slots=True)
class SingleModalityBaseline:
    """The best single-modality comparator the parity inequality is scored against."""

    modality: Modality

    @property
    def label(self) -> str:
        return f"single-modality {self.modality.value}"

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label, "modality": self.modality.value}


def default_design() -> SimulationDesign:
    design = SimulationDesign()
    design.validate()
    return design


def reference_cells() -> dict[str, SimulationCell]:
    """The two reference cells quoted in the manuscript's own narrative."""

    return {
        "moderate_conflict": SimulationCell(conflict_probability=0.3, absent_fraction=0.0),
        "absent_evidence": SimulationCell(conflict_probability=0.3, absent_fraction=0.25),
    }


def grid_axes() -> dict[str, tuple[float, ...]]:
    return {"conflict_probability": CONFLICT_GRID, "absent_fraction": ABSENT_GRID}
