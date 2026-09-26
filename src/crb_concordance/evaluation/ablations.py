"""The four-level extraction and the substitution tier of the ablation plan.

Ref: Sec. 4.5 (four levels of extraction: no-knowledge-graph, single-agent,
no-verifier and per-modality removal; and a substitution tier that keeps the four
modalities but replaces the combination rule with standard Dempster renormalization,
naive summation or averaging, and Bayesian log-odds fusion defined on ranks).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from crb_concordance.agents.mapping import MassMapper
from crb_concordance.belief.combine import (
    CombinationError,
    combine_modality_masses,
    dempster_sequential,
)
from crb_concordance.belief.discount import DiscountRates, discount
from crb_concordance.belief.fusion import bayesian_log_odds, naive_average, naive_sum
from crb_concordance.belief.mass import MassTriple, vacuous
from crb_concordance.cohorts.generator import SimulatedCandidate
from crb_concordance.metrics.ranking import rank_order, recall_at_k
from crb_concordance.simulation.evidence_model import evidence_from_candidate
from crb_concordance.utils.types import MODALITY_ORDER, Modality


class AblationError(ValueError):
    """Raised when an ablation level cannot be applied."""


class ExtractionLevel(str, Enum):
    FULL = "full"
    NO_KNOWLEDGE_GRAPH = "no-knowledge-graph"
    SINGLE_AGENT = "single-agent"
    NO_VERIFIER = "no-verifier"
    PER_MODALITY_REMOVAL = "per-modality-removal"


class SubstitutionRule(str, Enum):
    CONFLICT_REDISTRIBUTING = "conflict-redistributing"
    DEMPSTER_RENORMALISED = "dempster-renormalised"
    NAIVE_SUM = "naive-sum"
    NAIVE_AVERAGE = "naive-average"
    BAYESIAN_LOG_ODDS = "bayesian-log-odds"


@dataclass(frozen=True, slots=True)
class AblationOutcome:
    """One ablation cell with its ranking recall and interval statistics."""

    label: str
    level: str
    rule: str
    removed_modalities: tuple[str, ...]
    recall_at_k: float
    mean_interval_width: float
    mean_conflict: float
    mean_pignistic: float

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "level": self.level,
            "rule": self.rule,
            "removed_modalities": list(self.removed_modalities),
            "recall_at_k": self.recall_at_k,
            "mean_interval_width": self.mean_interval_width,
            "mean_conflict": self.mean_conflict,
            "mean_pignistic": self.mean_pignistic,
        }


def _masses(
    candidate: SimulatedCandidate,
    rates: DiscountRates,
    mapper: MassMapper,
    removed: tuple[Modality, ...],
) -> dict[Modality, MassTriple]:
    evidences = evidence_from_candidate(candidate)
    triples: dict[Modality, MassTriple] = {}
    for modality in MODALITY_ORDER:
        if modality in removed:
            triples[modality] = vacuous()
            continue
        triples[modality] = discount(mapper.map_evidence(evidences[modality]), rates.of(modality))
    return triples


@dataclass(frozen=True, slots=True)
class CellScores:
    """Ranking scores plus the concordance interval of the same cell."""

    scores: dict[str, float]
    widths: dict[str, float]
    conflicts: dict[str, float]


def score_pool(
    candidates: tuple[SimulatedCandidate, ...],
    rates: DiscountRates,
    *,
    level: ExtractionLevel = ExtractionLevel.FULL,
    rule: SubstitutionRule = SubstitutionRule.CONFLICT_REDISTRIBUTING,
    removed: tuple[Modality, ...] = (),
    single_modality: Modality = Modality.DEP,
    mapper: MassMapper | None = None,
) -> CellScores:
    """Ranking scores and concordance intervals for one ablation cell.

    The width and conflict columns always come from the conflict-redistributing
    mechanism on the cell's own modality set, so the interval statistics stay
    comparable across substitution rules that produce no interval of their own.
    """

    mass_mapper = mapper or MassMapper()
    effective_removed = tuple(removed)
    if level is ExtractionLevel.NO_KNOWLEDGE_GRAPH:
        effective_removed = tuple(sorted({*effective_removed, Modality.KG}, key=lambda m: m.value))
    scores: dict[str, float] = {}
    widths: dict[str, float] = {}
    conflicts: dict[str, float] = {}
    order = rates.reliability_order()
    for candidate in candidates:
        masses = _masses(candidate, rates, mass_mapper, effective_removed)
        reference = combine_modality_masses(masses, order)
        widths[candidate.gene] = reference.interval_width
        conflicts[candidate.gene] = reference.total_conflict
        if level is ExtractionLevel.SINGLE_AGENT:
            scores[candidate.gene] = masses[single_modality].pignistic()
            continue
        if (
            rule is SubstitutionRule.CONFLICT_REDISTRIBUTING
            and level is not ExtractionLevel.NO_VERIFIER
        ):
            scores[candidate.gene] = reference.pignistic
            continue
        if rule is SubstitutionRule.DEMPSTER_RENORMALISED:
            ordered = [masses[modality] for modality in order]
            try:
                combined = dempster_sequential(ordered, [modality.value for modality in order])
                scores[candidate.gene] = combined.pignistic
                widths[candidate.gene] = combined.interval_width
                conflicts[candidate.gene] = combined.total_conflict
            except CombinationError:
                scores[candidate.gene] = 0.5
                widths[candidate.gene] = 1.0
                conflicts[candidate.gene] = 1.0
            continue
        triples = [masses[modality] for modality in MODALITY_ORDER]
        if rule is SubstitutionRule.NAIVE_SUM:
            scores[candidate.gene] = naive_sum(triples).pignistic()
        elif rule is SubstitutionRule.NAIVE_AVERAGE:
            scores[candidate.gene] = naive_average(triples).pignistic()
        else:
            probabilities = {modality: masses[modality].pignistic() for modality in MODALITY_ORDER}
            scores[candidate.gene] = bayesian_log_odds(probabilities)
    return CellScores(scores=scores, widths=widths, conflicts=conflicts)


def evaluate_cell(
    candidates: tuple[SimulatedCandidate, ...],
    rates: DiscountRates,
    *,
    label: str,
    level: ExtractionLevel,
    rule: SubstitutionRule = SubstitutionRule.CONFLICT_REDISTRIBUTING,
    removed: tuple[Modality, ...] = (),
    single_modality: Modality = Modality.DEP,
    cutoff: int = 10,
    mapper: MassMapper | None = None,
) -> AblationOutcome:
    cell = score_pool(
        candidates,
        rates,
        level=level,
        rule=rule,
        removed=removed,
        single_modality=single_modality,
        mapper=mapper,
    )
    positives = {candidate.gene for candidate in candidates if candidate.vulnerable}
    return AblationOutcome(
        label=label,
        level=level.value,
        rule=rule.value,
        removed_modalities=tuple(modality.value for modality in removed),
        recall_at_k=recall_at_k(rank_order(cell.scores), positives, cutoff),
        mean_interval_width=float(np.mean(np.asarray(list(cell.widths.values()), dtype=float))),
        mean_conflict=float(np.mean(np.asarray(list(cell.conflicts.values()), dtype=float))),
        mean_pignistic=float(np.mean(np.asarray(list(cell.scores.values()), dtype=float))),
    )


def full_plan() -> dict[str, object]:
    """The declared ablation battery, as configuration rather than as results."""

    return {
        "extraction_levels": [level.value for level in ExtractionLevel],
        "per_modality_removals": [modality.value for modality in MODALITY_ORDER],
        "substitution_rules": [rule.value for rule in SubstitutionRule],
        "single_agent_default": Modality.DEP.value,
    }


def run_battery(
    candidates: tuple[SimulatedCandidate, ...],
    rates: DiscountRates,
    *,
    cutoff: int = 10,
    mapper: MassMapper | None = None,
) -> tuple[AblationOutcome, ...]:
    """Every declared ablation cell over one candidate pool."""

    if not candidates:
        raise AblationError("the candidate pool is empty")
    cells: list[AblationOutcome] = [
        evaluate_cell(
            candidates,
            rates,
            label="full",
            level=ExtractionLevel.FULL,
            cutoff=cutoff,
            mapper=mapper,
        ),
        evaluate_cell(
            candidates,
            rates,
            label="no-knowledge-graph",
            level=ExtractionLevel.NO_KNOWLEDGE_GRAPH,
            cutoff=cutoff,
            mapper=mapper,
        ),
        evaluate_cell(
            candidates,
            rates,
            label="no-verifier (naive average)",
            level=ExtractionLevel.NO_VERIFIER,
            rule=SubstitutionRule.NAIVE_AVERAGE,
            cutoff=cutoff,
            mapper=mapper,
        ),
    ]
    for modality in MODALITY_ORDER:
        cells.append(
            evaluate_cell(
                candidates,
                rates,
                label=f"single-agent ({modality.value})",
                level=ExtractionLevel.SINGLE_AGENT,
                single_modality=modality,
                cutoff=cutoff,
                mapper=mapper,
            )
        )
        cells.append(
            evaluate_cell(
                candidates,
                rates,
                label=f"remove {modality.value}",
                level=ExtractionLevel.PER_MODALITY_REMOVAL,
                removed=(modality,),
                cutoff=cutoff,
                mapper=mapper,
            )
        )
    for rule in (
        SubstitutionRule.DEMPSTER_RENORMALISED,
        SubstitutionRule.NAIVE_SUM,
        SubstitutionRule.NAIVE_AVERAGE,
        SubstitutionRule.BAYESIAN_LOG_ODDS,
    ):
        cells.append(
            evaluate_cell(
                candidates,
                rates,
                label=f"substitution: {rule.value}",
                level=ExtractionLevel.FULL,
                rule=rule,
                cutoff=cutoff,
                mapper=mapper,
            )
        )
    return tuple(cells)
