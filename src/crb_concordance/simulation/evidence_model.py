"""Turning simulated evidence into belief summaries through the shipped mechanism.

Ref: Sec. 4.1, Eq. (1)-(4) and Algorithm 2; the simulation reuses the same
combination engine and the same calibrated map as the discovery pass, so a change
in either is visible in the simulation contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.agents.base import RawEvidence
from crb_concordance.agents.mapping import MassMapper
from crb_concordance.agents.verifier import VerificationOutcome, Verifier
from crb_concordance.belief.discount import DiscountRates
from crb_concordance.belief.fusion import naive_average
from crb_concordance.cohorts.generator import SimulatedCandidate
from crb_concordance.utils.types import MODALITY_ORDER, Modality


class EvidenceModelError(ValueError):
    """Raised when a simulated candidate cannot be summarised."""


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """One simulated candidate's belief summary and hidden status."""

    gene: str
    vulnerable: bool
    pignistic: float
    interval_width: float
    total_conflict: float
    modality_pignistic: dict[Modality, float]
    absent_modalities: int
    flagged_steps: int
    trace_steps: int

    @property
    def trace_flagged(self) -> bool:
        return self.flagged_steps > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "gene": self.gene,
            "vulnerable": self.vulnerable,
            "pignistic": self.pignistic,
            "interval_width": self.interval_width,
            "total_conflict": self.total_conflict,
            "absent_modalities": self.absent_modalities,
            "flagged_steps": self.flagged_steps,
            "trace_steps": self.trace_steps,
        }


def evidence_from_candidate(candidate: SimulatedCandidate) -> dict[Modality, RawEvidence]:
    evidences: dict[Modality, RawEvidence] = {}
    for modality in MODALITY_ORDER:
        score = candidate.scores[modality]
        evidences[modality] = RawEvidence(
            candidate=candidate.gene,
            modality=modality,
            score=score,
            evidence_nature=(
                "simulated modality evidence" if score is not None else "simulated absent evidence"
            ),
            detail={"simulated": True},
        )
    return evidences


def summarise_candidate(
    candidate: SimulatedCandidate,
    verifier: Verifier,
) -> CandidateSummary:
    outcome: VerificationOutcome = verifier.combine_candidate(
        candidate.gene, evidence_from_candidate(candidate)
    )
    modality_pignistic = {
        modality: mass.triple.pignistic() for modality, mass in outcome.masses.items()
    }
    return CandidateSummary(
        gene=candidate.gene,
        vulnerable=candidate.vulnerable,
        pignistic=outcome.report.pignistic,
        interval_width=outcome.report.interval_width,
        total_conflict=outcome.combined.total_conflict,
        modality_pignistic=modality_pignistic,
        absent_modalities=sum(
            1 for modality in MODALITY_ORDER if candidate.scores[modality] is None
        ),
        flagged_steps=len(outcome.trace.flagged()),
        trace_steps=len(outcome.trace.entries),
    )


def summarise_pool(
    candidates: tuple[SimulatedCandidate, ...],
    rates: DiscountRates,
    *,
    mapper: MassMapper | None = None,
    conflict_threshold: float = 0.0,
) -> tuple[CandidateSummary, ...]:
    verifier = Verifier(
        rates=rates, mapper=mapper or MassMapper(), conflict_threshold=conflict_threshold
    )
    return tuple(summarise_candidate(candidate, verifier) for candidate in candidates)


def summarise_pool_naive(
    candidates: tuple[SimulatedCandidate, ...],
    rates: DiscountRates,
    *,
    mapper: MassMapper | None = None,
) -> tuple[CandidateSummary, ...]:
    """Re-score the same evidence with component-wise averaging instead of Algorithm 2.

    The naive variant carries no conflict trace, so its trace fields stay empty; that
    is exactly the difference the fusion-rule ablation is meant to expose.
    """

    from crb_concordance.belief.discount import discount

    mass_mapper = mapper or MassMapper()
    summaries: list[CandidateSummary] = []
    for candidate in candidates:
        discounted = {
            modality: discount(
                mass_mapper.map_evidence(evidence_from_candidate(candidate)[modality]),
                rates.of(modality),
            )
            for modality in MODALITY_ORDER
        }
        combined = naive_average(list(discounted.values()))
        modality_pignistic = {
            modality: triple.pignistic() for modality, triple in discounted.items()
        }
        summaries.append(
            CandidateSummary(
                gene=candidate.gene,
                vulnerable=candidate.vulnerable,
                pignistic=combined.pignistic(),
                interval_width=combined.interval_width(),
                total_conflict=0.0,
                modality_pignistic=modality_pignistic,
                absent_modalities=sum(
                    1 for modality in MODALITY_ORDER if candidate.scores[modality] is None
                ),
                flagged_steps=0,
                trace_steps=0,
            )
        )
    return tuple(summaries)


def pignistic_vector(summaries: tuple[CandidateSummary, ...]) -> np.ndarray:
    return np.asarray([summary.pignistic for summary in summaries], dtype=float)


def width_vector(summaries: tuple[CandidateSummary, ...]) -> np.ndarray:
    return np.asarray([summary.interval_width for summary in summaries], dtype=float)


def conflict_vector(summaries: tuple[CandidateSummary, ...]) -> np.ndarray:
    return np.asarray([summary.total_conflict for summary in summaries], dtype=float)


def label_vector(summaries: tuple[CandidateSummary, ...]) -> np.ndarray:
    return np.asarray([1.0 if summary.vulnerable else 0.0 for summary in summaries], dtype=float)


def modality_score_map(
    summaries: tuple[CandidateSummary, ...], modality: Modality
) -> dict[str, float]:
    return {summary.gene: summary.modality_pignistic[modality] for summary in summaries}


def best_single_modality(
    summaries: tuple[CandidateSummary, ...], *, cutoff: int
) -> tuple[Modality, float]:
    """The strongest single-modality comparator, chosen on the same pool."""

    from crb_concordance.metrics.ranking import rank_order, recall_at_k

    positives = {summary.gene for summary in summaries if summary.vulnerable}
    best_modality = MODALITY_ORDER[0]
    best_recall = -1.0
    for modality in MODALITY_ORDER:
        scores = modality_score_map(summaries, modality)
        recall = recall_at_k(rank_order(scores), positives, cutoff)
        if recall > best_recall:
            best_recall = recall
            best_modality = modality
    return best_modality, best_recall


def absent_only_summaries(summaries: tuple[CandidateSummary, ...]) -> tuple[CandidateSummary, ...]:
    return tuple(
        summary for summary in summaries if summary.absent_modalities == len(MODALITY_ORDER)
    )


def absent_only_width(summaries: tuple[CandidateSummary, ...]) -> float:
    subset = absent_only_summaries(summaries)
    if not subset:
        return 0.0
    return float(np.mean([summary.interval_width for summary in subset]))
