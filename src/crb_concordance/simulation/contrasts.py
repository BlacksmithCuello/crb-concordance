"""Contrasts, confidence intervals and the pre-specified hypothesis tests.

Ref: Sec. 4.4 (the pre-specified falsification inequality
recall@10(CRB-Concordance) - recall@10(best single-modality baseline) >= 15
percentage points); Sec. 4.5 (H1 on the fusion rule and H2 on the conflict trace,
whose requirement is that the interval-width partial R-squared collapses when the
trace is suppressed and exceeds the statistics-only value); Sec. 4.6 (seed-level
bootstrap with 5,000 resamples).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.agents.mapping import MassMapper
from crb_concordance.belief.discount import DiscountRates
from crb_concordance.metrics.comparison import seed_level_summary
from crb_concordance.metrics.ranking import rank_order, recall_at_k
from crb_concordance.simulation.design import SimulationDesign
from crb_concordance.simulation.evidence_model import (
    CandidateSummary,
    summarise_pool,
    summarise_pool_naive,
)
from crb_concordance.simulation.runner import CellResult, SimulationResult
from crb_concordance.utils.types import Verdict


class ContrastError(ValueError):
    """Raised when a contrast cannot be computed."""


@dataclass(frozen=True, slots=True)
class HypothesisOutcome:
    """One pre-specified hypothesis with its statistic and verdict."""

    name: str
    verdict: Verdict
    statistic: float
    threshold: float
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "statistic": self.statistic,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CellContrast:
    """Seed-level summary of one grid cell."""

    cell: str
    conflict_probability: float
    absent_fraction: float
    recall: dict[str, float]
    parity_gap: dict[str, float]
    conflict: dict[str, float]
    width_separability: dict[str, float]
    brier: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell,
            "conflict_probability": self.conflict_probability,
            "absent_fraction": self.absent_fraction,
            "recall": self.recall,
            "parity_gap": self.parity_gap,
            "conflict": self.conflict,
            "width_separability": self.width_separability,
            "brier": self.brier,
        }


def least_squares_r2(features: np.ndarray, target: np.ndarray) -> float:
    design = np.asarray(features, dtype=float)
    response = np.asarray(target, dtype=float)
    if design.ndim != 2:
        raise ContrastError("the design matrix must be two dimensional")
    if design.shape[0] != response.shape[0]:
        raise ContrastError("design matrix and target must share a row count")
    augmented = np.column_stack([np.ones(design.shape[0]), design])
    solution, *_ = np.linalg.lstsq(augmented, response, rcond=None)
    residual = response - augmented @ solution
    total = response - float(np.mean(response))
    denominator = float(total @ total)
    if denominator <= 1e-15:
        return 0.0
    return 1.0 - float(residual @ residual) / denominator


def partial_r2(
    full_features: np.ndarray, reduced_features: np.ndarray, target: np.ndarray
) -> float:
    """Incremental variance explained by the extra columns of ``full_features``."""

    full = least_squares_r2(full_features, target)
    reduced = least_squares_r2(reduced_features, target)
    denominator = 1.0 - reduced
    if denominator <= 1e-12:
        return 0.0
    return (full - reduced) / denominator


def trace_design_matrix(summaries: tuple[CandidateSummary, ...], *, with_trace: bool) -> np.ndarray:
    columns = [
        [summary.total_conflict for summary in summaries],
        [float(summary.absent_modalities) for summary in summaries],
    ]
    if with_trace:
        columns.extend(
            [
                [float(summary.flagged_steps) for summary in summaries],
                [float(summary.trace_steps) for summary in summaries],
                [1.0 if summary.trace_flagged else 0.0 for summary in summaries],
            ]
        )
    return np.column_stack(columns)


def conflict_trace_h2(
    summaries: tuple[CandidateSummary, ...],
) -> tuple[HypothesisOutcome, float, float, float]:
    """H2: the trace must add interval-width variance beyond the raw statistics."""

    target = np.asarray([summary.interval_width for summary in summaries], dtype=float)
    reduced = trace_design_matrix(summaries, with_trace=False)
    full = trace_design_matrix(summaries, with_trace=True)
    r2_reduced = least_squares_r2(reduced, target)
    r2_full = least_squares_r2(full, target)
    added = partial_r2(full, reduced, target)
    outcome = HypothesisOutcome(
        name="H2_conflict_trace_partial_r2",
        verdict=Verdict.PASS if added > 0.0 else Verdict.FAIL,
        statistic=added,
        threshold=0.0,
        detail=(
            f"interval-width R2 statistics only {r2_reduced:.6f}, with the trace {r2_full:.6f}, "
            f"partial R2 added by the trace {added:.6f}"
        ),
    )
    return outcome, added, r2_full, r2_reduced


def fusion_rule_h1(
    full: tuple[CandidateSummary, ...],
    naive: tuple[CandidateSummary, ...],
    *,
    cutoff: int,
    margin: float = 0.0,
) -> HypothesisOutcome:
    """H1: dropping the combination rule must cost recall on the same pool."""

    positives = {summary.gene for summary in full if summary.vulnerable}
    full_recall = recall_at_k(
        rank_order({summary.gene: summary.pignistic for summary in full}), positives, cutoff
    )
    naive_recall = recall_at_k(
        rank_order({summary.gene: summary.pignistic for summary in naive}), positives, cutoff
    )
    loss = full_recall - naive_recall
    return HypothesisOutcome(
        name="H1_fusion_rule_recall_loss",
        verdict=Verdict.PASS if loss > margin else Verdict.FAIL,
        statistic=loss,
        threshold=margin,
        detail=(
            f"recall@{cutoff} with the conflict-redistributing rule {full_recall:.4f} against "
            f"naive averaging {naive_recall:.4f}"
        ),
    )


def single_modality_recall(
    summaries: tuple[CandidateSummary, ...], modality: str, *, cutoff: int
) -> float:
    from crb_concordance.utils.types import Modality

    positives = {summary.gene for summary in summaries if summary.vulnerable}
    scores = {summary.gene: summary.modality_pignistic[Modality(modality)] for summary in summaries}
    return recall_at_k(rank_order(scores), positives, cutoff)


def cell_contrast(cell: CellResult, *, resamples: int, seed: int) -> CellContrast:
    runs = cell.runs
    return CellContrast(
        cell=cell.cell.label,
        conflict_probability=cell.cell.conflict_probability,
        absent_fraction=cell.cell.absent_fraction,
        recall=seed_level_summary(
            np.asarray([run.recall_at_k for run in runs], dtype=float),
            resamples=resamples,
            seed=seed,
        ),
        parity_gap=seed_level_summary(
            np.asarray([run.parity_gap for run in runs], dtype=float),
            resamples=resamples,
            seed=seed + 1,
        ),
        conflict=seed_level_summary(
            np.asarray([run.mean_conflict for run in runs], dtype=float),
            resamples=resamples,
            seed=seed + 2,
        ),
        width_separability=seed_level_summary(
            np.asarray([run.width_separability for run in runs], dtype=float),
            resamples=resamples,
            seed=seed + 3,
        ),
        brier=seed_level_summary(
            np.asarray([run.brier for run in runs], dtype=float),
            resamples=resamples,
            seed=seed + 4,
        ),
    )


def all_cell_contrasts(result: SimulationResult) -> tuple[CellContrast, ...]:
    return tuple(
        cell_contrast(cell, resamples=result.design.bootstrap_resamples, seed=100 + index)
        for index, cell in enumerate(result.cells)
    )


def falsification_inequality(result: SimulationResult) -> HypothesisOutcome:
    """The pre-specified recall@10 margin over the best single-modality baseline."""

    gaps = np.asarray([run.parity_gap for run in result.runs()], dtype=float)
    margin = result.design.parity_margin
    interval = seed_level_summary(gaps, resamples=result.design.bootstrap_resamples, seed=17)
    holds = interval["ci_lower"] >= margin
    return HypothesisOutcome(
        name="recall10_parity_inequality",
        verdict=Verdict.PASS if holds else Verdict.FAIL,
        statistic=interval["mean"],
        threshold=margin,
        detail=(
            f"mean parity gap over {gaps.size} runs {interval['mean']:.4f} "
            f"(95% CI {interval['ci_lower']:.4f}-{interval['ci_upper']:.4f}); "
            "the interval lower bound must clear the pre-specified margin"
        ),
    )


def falsification_inequality_saturated(result: SimulationResult) -> HypothesisOutcome:
    """The same margin measured at the cutoff where it is reachable at all."""

    gaps = np.asarray([run.parity_gap_at_saturated_cutoff for run in result.runs()], dtype=float)
    margin = result.design.parity_margin
    interval = seed_level_summary(gaps, resamples=result.design.bootstrap_resamples, seed=23)
    holds = interval["ci_lower"] >= margin
    cutoff = result.design.saturated_cutoff
    positives = result.design.cells[0].prevalence * result.design.cells[0].n_candidates
    return HypothesisOutcome(
        name="recall_parity_inequality_at_saturated_cutoff",
        verdict=Verdict.PASS if holds else Verdict.FAIL,
        statistic=interval["mean"],
        threshold=margin,
        detail=(
            f"cutoff {cutoff} gives a recall ceiling of "
            f"{cutoff / positives:.4f} against the declared pool of "
            f"{int(positives)} positives; mean parity gap {interval['mean']:.4f} "
            f"(95% CI {interval['ci_lower']:.4f}-{interval['ci_upper']:.4f})"
        ),
    )


def cutoff_reachability(result: SimulationResult) -> dict[str, float]:
    """How far the declared margin is from being reachable at the declared cutoff."""

    first = result.design.cells[0]
    positives = first.prevalence * first.n_candidates
    ceiling = result.design.recall_cutoff / positives
    return {
        "declared_cutoff": float(result.design.recall_cutoff),
        "positives_per_pool": float(positives),
        "recall_ceiling_at_declared_cutoff": ceiling,
        "declared_margin": result.design.parity_margin,
        "margin_reachable_at_declared_cutoff": (
            1.0 if ceiling >= result.design.parity_margin else 0.0
        ),
        "first_reachable_cutoff": float(result.design.saturated_cutoff),
    }


def conflict_monotonicity_across_cells(result: SimulationResult) -> HypothesisOutcome:
    """Conflict mass must rise with the declared conflict probability."""

    by_conflict: dict[float, list[float]] = {}
    for cell in result.cells:
        by_conflict.setdefault(cell.cell.conflict_probability, []).extend(
            [run.mean_conflict for run in cell.runs]
        )
    levels = sorted(by_conflict)
    means = {level: float(np.mean(by_conflict[level])) for level in levels}
    ordered = [means[level] for level in levels]
    monotone = all(
        ordered[index] <= ordered[index + 1] + 1e-12 for index in range(len(ordered) - 1)
    )
    return HypothesisOutcome(
        name="conflict_mass_increases_with_declared_conflict",
        verdict=Verdict.PASS if monotone else Verdict.FAIL,
        statistic=float(ordered[-1] - ordered[0]) if ordered else 0.0,
        threshold=0.0,
        detail=f"mean conflict by declared conflict probability {means}",
    )


def absent_evidence_widens_intervals(result: SimulationResult) -> HypothesisOutcome:
    """More absent evidence must not narrow the intervals."""

    by_absent: dict[float, list[float]] = {}
    for cell in result.cells:
        by_absent.setdefault(cell.cell.absent_fraction, []).extend(
            [run.mean_interval_width for run in cell.runs]
        )
    levels = sorted(by_absent)
    means = {level: float(np.mean(by_absent[level])) for level in levels}
    ordered = [means[level] for level in levels]
    monotone = all(
        ordered[index] <= ordered[index + 1] + 1e-12 for index in range(len(ordered) - 1)
    )
    return HypothesisOutcome(
        name="interval_width_increases_with_absent_evidence",
        verdict=Verdict.PASS if monotone else Verdict.FAIL,
        statistic=float(ordered[-1] - ordered[0]) if ordered else 0.0,
        threshold=0.0,
        detail=f"mean width by absent fraction {means}",
    )


def pooled_trace_analysis(
    rates: DiscountRates,
    design: SimulationDesign,
    *,
    mapper: MassMapper | None = None,
    seeds_per_cell: int = 5,
    flag_threshold: float = 0.05,
) -> dict[str, object]:
    """H1 and H2 evaluated on a pooled sample drawn from every grid cell."""

    mass_mapper = mapper or MassMapper()
    pooled: list[CandidateSummary] = []
    naive: list[CandidateSummary] = []
    for cell in design.cells:
        for seed in cell.seeds[:seeds_per_cell]:
            candidates = cell.candidates(seed)
            pooled.extend(
                summarise_pool(
                    candidates, rates, mapper=mass_mapper, conflict_threshold=flag_threshold
                )
            )
            naive.extend(summarise_pool_naive(candidates, rates, mapper=mass_mapper))
    outcome_h2, added, r2_full, r2_reduced = conflict_trace_h2(tuple(pooled))
    outcome_h1 = fusion_rule_h1(
        tuple(pooled), tuple(naive), cutoff=design.recall_cutoff, margin=0.0
    )
    return {
        "candidates": len(pooled),
        "flag_threshold": flag_threshold,
        "h1": outcome_h1.as_dict(),
        "h2": outcome_h2.as_dict(),
        "h2_partial_r2_added": added,
        "h2_r2_with_trace": r2_full,
        "h2_r2_statistics_only": r2_reduced,
        "single_modality_recall": {
            modality: single_modality_recall(tuple(pooled), modality, cutoff=design.recall_cutoff)
            for modality in ("KG", "DEP", "FLUX", "CLIN")
        },
    }
