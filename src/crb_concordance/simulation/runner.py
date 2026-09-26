"""Execution of the simulation grid.

Ref: Sec. 2 (the in silico simulation quantifies the conflict distribution over the
declared synthetic evidence parameters); Sec. 4.4 (the pre-specified recall@10
falsification inequality); Sec. 4.5 (the fusion and conflict-trace ablations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from crb_concordance.agents.mapping import MassMapper
from crb_concordance.belief.discount import DiscountRates
from crb_concordance.metrics.calibration import brier_score, expected_calibration_error
from crb_concordance.metrics.ranking import (
    mean_reciprocal_rank,
    precision_at_k,
    rank_order,
    recall_at_k,
)
from crb_concordance.simulation.design import SimulationCell, SimulationDesign
from crb_concordance.simulation.evidence_model import (
    CandidateSummary,
    absent_only_width,
    best_single_modality,
    summarise_pool,
)
from crb_concordance.utils.types import Modality


class RunnerError(ValueError):
    """Raised when a simulation run cannot be completed."""


@dataclass(frozen=True, slots=True)
class SeedRun:
    """Every quantity one (cell, seed) run reports."""

    cell: str
    seed: int
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    best_single_recall: float
    best_single_modality: str
    parity_gap: float
    mean_conflict: float
    mean_interval_width: float
    width_separability: float
    brier: float
    expected_calibration_error: float
    absent_only_width: float
    recall_at_saturated_cutoff: float
    best_single_recall_at_saturated_cutoff: float
    parity_gap_at_saturated_cutoff: float

    def as_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell,
            "seed": self.seed,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "best_single_recall": self.best_single_recall,
            "best_single_modality": self.best_single_modality,
            "parity_gap": self.parity_gap,
            "mean_conflict": self.mean_conflict,
            "mean_interval_width": self.mean_interval_width,
            "width_separability": self.width_separability,
            "brier": self.brier,
            "expected_calibration_error": self.expected_calibration_error,
            "absent_only_width": self.absent_only_width,
            "recall_at_saturated_cutoff": self.recall_at_saturated_cutoff,
            "best_single_recall_at_saturated_cutoff": self.best_single_recall_at_saturated_cutoff,
            "parity_gap_at_saturated_cutoff": self.parity_gap_at_saturated_cutoff,
        }


def cross_modal_disagreement(summary: CandidateSummary) -> float:
    """Spread of the four modality pignistic values for one candidate."""

    values = np.asarray(list(summary.modality_pignistic.values()), dtype=float)
    return float(np.max(values) - np.min(values))


def concordance_labels(summaries: tuple[CandidateSummary, ...], *, margin: float) -> np.ndarray:
    """Concordant when every modality pair agrees within ``margin``.

    The label is defined on cross-modality agreement, not on the width, so scoring the
    width against it is a genuine test rather than a restatement of the width.
    """

    return np.asarray(
        [1.0 if cross_modal_disagreement(summary) <= margin else 0.0 for summary in summaries],
        dtype=float,
    )


def width_separability(summaries: tuple[CandidateSummary, ...], *, margin: float) -> float:
    """AUROC of the interval width for separating concordant from discordant profiles."""

    widths = np.asarray([summary.interval_width for summary in summaries], dtype=float)
    labels = concordance_labels(summaries, margin=margin)
    positives = int(np.count_nonzero(labels > 0.5))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = rankdata(-widths)
    rank_sum = float(np.sum(ranks[labels > 0.5]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def run_seed(
    cell: SimulationCell,
    seed: int,
    rates: DiscountRates,
    design: SimulationDesign,
    *,
    mapper: MassMapper | None = None,
) -> SeedRun:
    candidates = cell.candidates(seed)
    summaries = summarise_pool(candidates, rates, mapper=mapper)
    positives = {summary.gene for summary in summaries if summary.vulnerable}
    if not positives:
        raise RunnerError(f"seed {seed} produced no positive candidates")
    combined_scores = {summary.gene: summary.pignistic for summary in summaries}
    ranked = rank_order(combined_scores)
    modality, single_recall = best_single_modality(summaries, cutoff=design.recall_cutoff)
    combined_recall = recall_at_k(ranked, positives, design.recall_cutoff)
    saturated = design.saturated_cutoff
    _, single_saturated = best_single_modality(summaries, cutoff=saturated)
    combined_saturated = recall_at_k(ranked, positives, saturated)
    probabilities = np.asarray([summary.pignistic for summary in summaries], dtype=float)
    labels = np.asarray([1.0 if summary.vulnerable else 0.0 for summary in summaries], dtype=float)
    return SeedRun(
        cell=cell.label,
        seed=seed,
        recall_at_k=combined_recall,
        precision_at_k=precision_at_k(ranked, positives, design.recall_cutoff),
        mean_reciprocal_rank=mean_reciprocal_rank(ranked, positives),
        best_single_recall=single_recall,
        best_single_modality=modality.value,
        parity_gap=combined_recall - single_recall,
        mean_conflict=float(np.mean([summary.total_conflict for summary in summaries])),
        mean_interval_width=float(np.mean([summary.interval_width for summary in summaries])),
        width_separability=width_separability(summaries, margin=design.concordance_margin),
        brier=brier_score(probabilities, labels),
        expected_calibration_error=expected_calibration_error(probabilities, labels),
        absent_only_width=absent_only_width(summaries),
        recall_at_saturated_cutoff=combined_saturated,
        best_single_recall_at_saturated_cutoff=single_saturated,
        parity_gap_at_saturated_cutoff=combined_saturated - single_saturated,
    )


@dataclass(frozen=True, slots=True)
class CellResult:
    cell: SimulationCell
    runs: tuple[SeedRun, ...]

    def recalls(self) -> np.ndarray:
        return np.asarray([run.recall_at_k for run in self.runs], dtype=float)

    def parity_gaps(self) -> np.ndarray:
        return np.asarray([run.parity_gap for run in self.runs], dtype=float)

    def conflicts(self) -> np.ndarray:
        return np.asarray([run.mean_conflict for run in self.runs], dtype=float)

    def separabilities(self) -> np.ndarray:
        return np.asarray([run.width_separability for run in self.runs], dtype=float)

    def as_dict(self) -> dict[str, object]:
        return {"cell": self.cell.as_dict(), "runs": [run.as_dict() for run in self.runs]}


@dataclass(frozen=True, slots=True)
class SimulationResult:
    design: SimulationDesign
    cells: tuple[CellResult, ...]
    rates: DiscountRates
    details: dict[str, object] = field(default_factory=dict)

    def runs(self) -> tuple[SeedRun, ...]:
        return tuple(run for cell in self.cells for run in cell.runs)

    def as_dict(self) -> dict[str, object]:
        return {
            "design": self.design.as_dict(),
            "rates": self.rates.as_dict(),
            "ignorance_floor": self.rates.ignorance_floor(),
            "cells": [cell.as_dict() for cell in self.cells],
            "details": dict(self.details),
        }


def run_cell(
    cell: SimulationCell,
    rates: DiscountRates,
    design: SimulationDesign,
    *,
    mapper: MassMapper | None = None,
) -> CellResult:
    runs = tuple(run_seed(cell, seed, rates, design, mapper=mapper) for seed in cell.seeds)
    return CellResult(cell=cell, runs=runs)


def run_design(
    rates: DiscountRates,
    *,
    design: SimulationDesign | None = None,
    mapper: MassMapper | None = None,
    cells: tuple[SimulationCell, ...] | None = None,
) -> SimulationResult:
    settings = design or SimulationDesign()
    settings.validate()
    selected = cells if cells is not None else settings.cells
    results = tuple(run_cell(cell, rates, settings, mapper=mapper) for cell in selected)
    return SimulationResult(
        design=settings,
        cells=results,
        rates=rates,
        details={
            "recall_cutoff": settings.recall_cutoff,
            "concordance_margin": settings.concordance_margin,
            "parity_margin": settings.parity_margin,
            "modalities": [modality.value for modality in Modality],
        },
    )
