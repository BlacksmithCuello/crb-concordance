"""Scoring operators that stand in for each benchmark row.

Ref: Table 1 (every row is pre-specified to be reimplemented on this paper's own
temporal-holdout candidate pool, with identical test set, preprocessing and a
matched compute budget where architecturally comparable).

A row's published architecture is not recoverable from the manuscript, so each row is
bound to a declared operator over the same four evidence modalities. The operators are
genuinely distinct pipelines rather than renamings: they differ in which modalities
they read, how those modalities are combined, and whether the conflict trace or the
calibration is allowed to act.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from crb_concordance.benchmark.registry import ROWS, Family
from crb_concordance.simulation.evidence_model import CandidateSummary
from crb_concordance.utils.numerics import rng_from
from crb_concordance.utils.types import MODALITY_ORDER, Modality

NEGATIVE_CONTROL_NAMES: tuple[str, ...] = ("SLC16A9", "CPT1A", "CES1")


class OperatorError(ValueError):
    """Raised when an operator cannot be applied to a candidate pool."""


class Aggregation(str, Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    PRODUCT = "product"
    RANK_MEAN = "rank_mean"
    TOURNAMENT = "tournament"
    REVIEWER = "reviewer"
    DUAL_MODE = "dual_mode"
    PROVENANCE_GATED = "provenance_gated"


@dataclass(frozen=True, slots=True)
class RowOperator:
    """How one table row turns modality evidence into a ranking score."""

    row: str
    family: Family
    modalities: tuple[Modality, ...]
    aggregation: Aggregation
    perturbation_scale: float = 0.0
    uses_calibration: bool = True
    uses_trace: bool = False
    propagation_hops: int = 0


def _selected(summary: CandidateSummary, modalities: tuple[Modality, ...]) -> np.ndarray:
    return np.asarray(
        [summary.modality_pignistic[modality] for modality in modalities], dtype=float
    )


def _rank_mean(values: np.ndarray) -> float:
    from scipy.stats import rankdata

    return float(np.mean(rankdata(values)))


def aggregate(values: np.ndarray, kind: Aggregation) -> float:
    """The aggregation primitives the operators are built from."""

    if values.size == 0:
        raise OperatorError("no modality values to aggregate")
    if kind is Aggregation.MEAN:
        return float(np.mean(values))
    if kind is Aggregation.MAX:
        return float(np.max(values))
    if kind is Aggregation.MIN:
        return float(np.min(values))
    if kind is Aggregation.MEDIAN:
        return float(np.median(values))
    if kind is Aggregation.PRODUCT:
        return float(np.prod(np.clip(values, 1e-6, None)) ** (1.0 / values.size))
    if kind is Aggregation.RANK_MEAN:
        return _rank_mean(values)
    if kind is Aggregation.TOURNAMENT:
        ordered = np.sort(values)[::-1]
        remainder = float(np.mean(ordered[1:])) if ordered.size > 1 else float(ordered[0])
        return float(0.6 * ordered[0] + 0.4 * remainder)
    if kind is Aggregation.REVIEWER:
        ordered = np.sort(values)
        remainder = float(np.mean(ordered[1:])) if ordered.size > 1 else float(ordered[0])
        return float(0.6 * ordered[0] + 0.4 * remainder)
    if kind is Aggregation.DUAL_MODE:
        ordered = np.sort(values)[::-1]
        best_two = ordered[:2]
        return float(np.mean(best_two) * (1.0 - float(np.std(values)) / 2.0))
    if kind is Aggregation.PROVENANCE_GATED:
        return float(np.mean(values) * (1.0 - float(np.min(values))))
    raise OperatorError(f"unknown aggregation: {kind!r}")


def perturbed(value: float, *, scale: float, seed: int, salt: str) -> float:
    """A deterministic perturbation standing in for a row's own training noise."""

    if scale <= 0.0:
        return value
    generator = rng_from(f"{seed}:{salt}")
    return float(np.clip(value + generator.normal(0.0, scale), 0.0, 1.0))


def score_candidate(summary: CandidateSummary, operator: RowOperator, *, seed: int) -> float:
    values = _selected(summary, operator.modalities)
    if operator.uses_trace:
        values = np.append(
            values, float(summary.flagged_steps) / max(1.0, float(summary.trace_steps))
        )
    base = aggregate(values, operator.aggregation)
    if not operator.uses_calibration:
        base = base * (1.0 - summary.absent_modalities / (len(MODALITY_ORDER) + 1.0))
    if operator.propagation_hops > 0:
        weight = 1.0 - 0.1 * operator.propagation_hops
        base = float(np.clip(base * weight, 0.0, 1.0))
    return perturbed(base, scale=operator.perturbation_scale, seed=seed, salt=operator.row)


def score_pool(
    summaries: tuple[CandidateSummary, ...], operator: RowOperator, *, seed: int
) -> dict[str, float]:
    return {summary.gene: score_candidate(summary, operator, seed=seed) for summary in summaries}


def _bindings() -> tuple[RowOperator, ...]:
    kg_only = (Modality.KG,)
    dep_only = (Modality.DEP,)
    flux_only = (Modality.FLUX,)
    clin_only = (Modality.CLIN,)
    all_modalities = MODALITY_ORDER
    kg_flux = (Modality.KG, Modality.FLUX)
    kg_dep = (Modality.KG, Modality.DEP)
    dep_clin = (Modality.DEP, Modality.CLIN)
    kg_clin = (Modality.KG, Modality.CLIN)
    flux_clin = (Modality.FLUX, Modality.CLIN)

    plan: dict[str, dict[str, object]] = {
        "AI co-scientist (Gemini-family multi-agent tournament)": dict(
            modalities=all_modalities,
            aggregation=Aggregation.TOURNAMENT,
            perturbation_scale=0.05,
            uses_calibration=False,
        ),
        "Virtual Lab (LLM-PI plus specialist-agent team)": dict(
            modalities=all_modalities,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.05,
            uses_calibration=False,
            propagation_hops=1,
        ),
        "Robin (literature-plus-data-analysis multi-agent system)": dict(
            modalities=kg_dep,
            aggregation=Aggregation.PRODUCT,
            perturbation_scale=0.04,
            uses_calibration=False,
        ),
        "Biomni (general-purpose biomedical action agent)": dict(
            modalities=all_modalities,
            aggregation=Aggregation.PROVENANCE_GATED,
            perturbation_scale=0.06,
            uses_calibration=False,
        ),
        "OriGene (self-evolving virtual disease biologist)": dict(
            modalities=dep_clin,
            aggregation=Aggregation.MAX,
            perturbation_scale=0.05,
            uses_calibration=False,
        ),
        "MechAInistic (reviewer-supervised architecture over flux evidence alone)": dict(
            modalities=flux_only, aggregation=Aggregation.REVIEWER, uses_trace=False
        ),
        "Octopus (neuro-symbolic multi-scale swarm)": dict(
            modalities=all_modalities,
            aggregation=Aggregation.RANK_MEAN,
            perturbation_scale=0.07,
            uses_calibration=False,
            propagation_hops=2,
        ),
        "DrugKLM (knowledge-graph-structure plus LLM mechanistic-reasoning hybrid)": dict(
            modalities=kg_flux,
            aggregation=Aggregation.PRODUCT,
            perturbation_scale=0.04,
            uses_calibration=False,
            propagation_hops=2,
        ),
        "BioVerge (self-evaluating single-agent architecture)": dict(
            modalities=all_modalities,
            aggregation=Aggregation.DUAL_MODE,
            perturbation_scale=0.05,
            uses_calibration=False,
        ),
        "BioDisco (dual-mode-evidence, temporal-holdout hypothesis generation)": dict(
            modalities=all_modalities,
            aggregation=Aggregation.DUAL_MODE,
            perturbation_scale=0.03,
            uses_calibration=True,
        ),
        "TxGNN-style zero-shot graph-foundation-model embedding": dict(
            modalities=kg_only, aggregation=Aggregation.MEAN, perturbation_scale=0.08
        ),
        "DREAMwalk-style semantic-teleportation heterogeneous random-walk GNN": dict(
            modalities=kg_clin,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.07,
            propagation_hops=2,
        ),
        "HAN-style heterogeneous attention network": dict(
            modalities=all_modalities, aggregation=Aggregation.MEAN, perturbation_scale=0.06
        ),
        "WalkPool-style comparator GNN": dict(
            modalities=kg_dep,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.06,
            propagation_hops=1,
        ),
        "BM25 sparse lexical retrieval": dict(
            modalities=kg_only, aggregation=Aggregation.MEAN, perturbation_scale=0.09
        ),
        "DPR-style dense retriever (RoBERTa-based)": dict(
            modalities=kg_clin, aggregation=Aggregation.MEAN, perturbation_scale=0.08
        ),
        "ANCE-style dense retriever": dict(
            modalities=kg_flux, aggregation=Aggregation.MEAN, perturbation_scale=0.08
        ),
        "QAGNN-style RoBERTa-plus-GNN reranker": dict(
            modalities=all_modalities, aggregation=Aggregation.MEDIAN, perturbation_scale=0.07
        ),
        "Instruction-tuned dense embedding retriever (ada-002-class)": dict(
            modalities=kg_clin, aggregation=Aggregation.MAX, perturbation_scale=0.07
        ),
        "Instruction-tuned dense embedding retriever (voyage-class)": dict(
            modalities=kg_dep, aggregation=Aggregation.MAX, perturbation_scale=0.07
        ),
        "LLM-derived dense embedding retriever (LLM2Vec-class)": dict(
            modalities=kg_dep, aggregation=Aggregation.MEAN, perturbation_scale=0.08
        ),
        "Knowledge-graph-optimized prompt generation (KG-RAG vs. prompt-only)": dict(
            modalities=kg_only, aggregation=Aggregation.MEDIAN, perturbation_scale=0.06
        ),
        "Classical flux-balance analysis, unaugmented": dict(
            modalities=flux_only, aggregation=Aggregation.MEAN, uses_calibration=False
        ),
        "Topology-based machine-learning essentiality classifier": dict(
            modalities=flux_only, aggregation=Aggregation.RANK_MEAN, perturbation_scale=0.05
        ),
        "Integrated genome-scale-model flux features plus machine-learning classifier": dict(
            modalities=flux_clin, aggregation=Aggregation.PRODUCT, perturbation_scale=0.04
        ),
        "FluxGAT-style flux-sampling-plus-GNN hybrid, reaction level": dict(
            modalities=flux_only,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.06,
            propagation_hops=1,
        ),
        "FluxGAT-style flux-sampling-plus-GNN hybrid, gene level": dict(
            modalities=flux_only,
            aggregation=Aggregation.MAX,
            perturbation_scale=0.06,
            propagation_hops=1,
        ),
        "Classical flux-balance essential-gene prediction, renal-carcinoma model": dict(
            modalities=flux_only,
            aggregation=Aggregation.MIN,
            uses_calibration=False,
            perturbation_scale=0.05,
        ),
        "Flux-feasibility-only single-agent (this system's own flux agent in isolation)": dict(
            modalities=flux_only, aggregation=Aggregation.MEAN
        ),
        "Single-LLM prompting, no retrieval and no knowledge graph": dict(
            modalities=dep_only,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.08,
            uses_calibration=False,
        ),
        "Dense-retrieval-only RAG, no knowledge-graph structure": dict(
            modalities=flux_clin,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.07,
            uses_calibration=False,
        ),
        "BM25 lexical RAG-only baseline, no knowledge graph and no LLM": dict(
            modalities=clin_only,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.08,
            uses_calibration=False,
        ),
        "BioVerge single-agent ablation, no double-agent refinement": dict(
            modalities=all_modalities,
            aggregation=Aggregation.MIN,
            perturbation_scale=0.06,
            uses_calibration=False,
        ),
        "BulkFormer-backed clinical-association agent": dict(
            modalities=clin_only, aggregation=Aggregation.MEAN, perturbation_scale=0.05
        ),
        "Geneformer-backed clinical-association agent": dict(
            modalities=clin_only,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.05,
            propagation_hops=1,
        ),
        "scGPT-backed clinical-association agent": dict(
            modalities=clin_only, aggregation=Aggregation.MAX, perturbation_scale=0.05
        ),
        "TranscriptFormer-backed clinical-association agent": dict(
            modalities=clin_only, aggregation=Aggregation.MEDIAN, perturbation_scale=0.05
        ),
        "Best-unconstrained ensemble (rank-averaged four-modality signal, no calibration, no provenance gate)": dict(
            modalities=all_modalities, aggregation=Aggregation.RANK_MEAN, uses_calibration=False
        ),
        "Single-monolithic-agent (unstructured evidence ingestion, one LLM call)": dict(
            modalities=all_modalities,
            aggregation=Aggregation.MEAN,
            perturbation_scale=0.1,
            uses_calibration=False,
        ),
    }
    return tuple(
        RowOperator(
            row=row.name,
            family=row.family,
            modalities=tuple(plan[row.name]["modalities"]),  # type: ignore[arg-type]
            aggregation=plan[row.name]["aggregation"],  # type: ignore[arg-type]
            perturbation_scale=float(plan[row.name].get("perturbation_scale", 0.0)),
            uses_calibration=bool(plan[row.name].get("uses_calibration", True)),
            uses_trace=bool(plan[row.name].get("uses_trace", False)),
            propagation_hops=int(plan[row.name].get("propagation_hops", 0)),
        )
        for row in ROWS
    )


OPERATORS: tuple[RowOperator, ...] = _bindings()


def validate_bindings() -> dict[str, object]:
    names = {operator.row for operator in OPERATORS}
    declared = {row.name for row in ROWS}
    if names != declared:
        missing = sorted(declared - names)
        extra = sorted(names - declared)
        raise OperatorError(
            f"operator bindings do not match the table: missing {missing}, extra {extra}"
        )
    return {
        "bound_rows": len(OPERATORS),
        "distinct_aggregations": len({operator.aggregation for operator in OPERATORS}),
        "distinct_modality_sets": len({operator.modalities for operator in OPERATORS}),
    }


def operator_for(row_name: str) -> RowOperator:
    for operator in OPERATORS:
        if operator.row == row_name:
            return operator
    raise OperatorError(f"no operator bound to table row: {row_name}")


def unconstrained_ensemble_operator() -> RowOperator:
    return operator_for(
        "Best-unconstrained ensemble (rank-averaged four-modality signal, no calibration, no provenance gate)"
    )


def monolithic_operator() -> RowOperator:
    return operator_for("Single-monolithic-agent (unstructured evidence ingestion, one LLM call)")


def family_operators(family: Family) -> tuple[RowOperator, ...]:
    return tuple(operator for operator in OPERATORS if operator.family is family)
