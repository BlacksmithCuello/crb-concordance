"""Transcription of the temporal-holdout rediscovery benchmark table.

Ref: Table 1 (37 agent, knowledge-graph and mechanistic-modelling baselines across
six architectural families, plus two fusion-mechanism controls, and the proposed
full system; every row pre-specified for reimplementation is marked, every outcome
cell is pending, and the table is split into a primary-utility panel and a
constraint-axis panel).

The rows, families, counts and reference numbers are transcribed exactly; the
operators that stand in for each row's scoring pipeline are declared engineering
defaults, because the manuscript specifies the rows rather than their
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

FUSION_PROPOSED = "CRB-Concordance (proposed, full system)"


class Family(str, Enum):
    MULTI_AGENT = "multi-agent LLM hypothesis generation"
    GRAPH_EMBEDDING = "GNN / knowledge-graph-embedding target ranking"
    RETRIEVAL = "RAG-LLM / retrieval-only knowledge-graph baseline"
    MECHANISTIC = "mechanistic metabolic modeling, single-modality"
    SINGLE_LLM = "single-LLM / RAG-only ablation controls"
    TRANSCRIPTOMIC = "transcriptomic foundation-model backbones for the clinical-association agent"
    FUSION_CONTROL = "fusion-mechanism controls"


PRIMARY_COLUMNS: tuple[str, ...] = ("Recall@10", "Prec.@10", "MRR")
CONSTRAINT_COLUMNS: tuple[str, ...] = (
    "Hallucination-rate (%)",
    "ECE",
    "Brier",
    "Prov. (%)",
)


@dataclass(frozen=True, slots=True)
class BaselineRow:
    """One row of Table 1."""

    name: str
    family: Family
    reference: int | None
    reimplemented: bool

    @property
    def flagged(self) -> str:
        return "reimplemented" if self.reimplemented else "original benchmark only"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family.value,
            "reference": self.reference,
            "reimplemented": self.reimplemented,
        }


ROWS: tuple[BaselineRow, ...] = (
    BaselineRow(
        "AI co-scientist (Gemini-family multi-agent tournament)", Family.MULTI_AGENT, 7, True
    ),
    BaselineRow("Virtual Lab (LLM-PI plus specialist-agent team)", Family.MULTI_AGENT, 9, True),
    BaselineRow(
        "Robin (literature-plus-data-analysis multi-agent system)", Family.MULTI_AGENT, 8, True
    ),
    BaselineRow("Biomni (general-purpose biomedical action agent)", Family.MULTI_AGENT, 10, True),
    BaselineRow("OriGene (self-evolving virtual disease biologist)", Family.MULTI_AGENT, 11, True),
    BaselineRow(
        "MechAInistic (reviewer-supervised architecture over flux evidence alone)",
        Family.MULTI_AGENT,
        12,
        True,
    ),
    BaselineRow("Octopus (neuro-symbolic multi-scale swarm)", Family.MULTI_AGENT, 13, True),
    BaselineRow(
        "DrugKLM (knowledge-graph-structure plus LLM mechanistic-reasoning hybrid)",
        Family.MULTI_AGENT,
        16,
        True,
    ),
    BaselineRow(
        "BioVerge (self-evaluating single-agent architecture)", Family.MULTI_AGENT, 17, True
    ),
    BaselineRow(
        "BioDisco (dual-mode-evidence, temporal-holdout hypothesis generation)",
        Family.MULTI_AGENT,
        15,
        True,
    ),
    BaselineRow(
        "TxGNN-style zero-shot graph-foundation-model embedding", Family.GRAPH_EMBEDDING, 21, True
    ),
    BaselineRow(
        "DREAMwalk-style semantic-teleportation heterogeneous random-walk GNN",
        Family.GRAPH_EMBEDDING,
        28,
        True,
    ),
    BaselineRow("HAN-style heterogeneous attention network", Family.GRAPH_EMBEDDING, 29, True),
    BaselineRow("WalkPool-style comparator GNN", Family.GRAPH_EMBEDDING, 30, True),
    BaselineRow("BM25 sparse lexical retrieval", Family.RETRIEVAL, None, True),
    BaselineRow("DPR-style dense retriever (RoBERTa-based)", Family.RETRIEVAL, None, True),
    BaselineRow("ANCE-style dense retriever", Family.RETRIEVAL, None, True),
    BaselineRow("QAGNN-style RoBERTa-plus-GNN reranker", Family.RETRIEVAL, None, True),
    BaselineRow(
        "Instruction-tuned dense embedding retriever (ada-002-class)", Family.RETRIEVAL, None, True
    ),
    BaselineRow(
        "Instruction-tuned dense embedding retriever (voyage-class)", Family.RETRIEVAL, None, True
    ),
    BaselineRow(
        "LLM-derived dense embedding retriever (LLM2Vec-class)", Family.RETRIEVAL, None, True
    ),
    BaselineRow(
        "Knowledge-graph-optimized prompt generation (KG-RAG vs. prompt-only)",
        Family.RETRIEVAL,
        20,
        True,
    ),
    BaselineRow("Classical flux-balance analysis, unaugmented", Family.MECHANISTIC, 63, True),
    BaselineRow(
        "Topology-based machine-learning essentiality classifier", Family.MECHANISTIC, None, True
    ),
    BaselineRow(
        "Integrated genome-scale-model flux features plus machine-learning classifier",
        Family.MECHANISTIC,
        35,
        True,
    ),
    BaselineRow(
        "FluxGAT-style flux-sampling-plus-GNN hybrid, reaction level", Family.MECHANISTIC, 37, True
    ),
    BaselineRow(
        "FluxGAT-style flux-sampling-plus-GNN hybrid, gene level", Family.MECHANISTIC, 37, True
    ),
    BaselineRow(
        "Classical flux-balance essential-gene prediction, renal-carcinoma model",
        Family.MECHANISTIC,
        36,
        True,
    ),
    BaselineRow(
        "Flux-feasibility-only single-agent (this system's own flux agent in isolation)",
        Family.MECHANISTIC,
        None,
        False,
    ),
    BaselineRow(
        "Single-LLM prompting, no retrieval and no knowledge graph", Family.SINGLE_LLM, None, True
    ),
    BaselineRow(
        "Dense-retrieval-only RAG, no knowledge-graph structure", Family.SINGLE_LLM, None, True
    ),
    BaselineRow(
        "BM25 lexical RAG-only baseline, no knowledge graph and no LLM",
        Family.SINGLE_LLM,
        None,
        True,
    ),
    BaselineRow(
        "BioVerge single-agent ablation, no double-agent refinement", Family.SINGLE_LLM, 17, True
    ),
    BaselineRow("BulkFormer-backed clinical-association agent", Family.TRANSCRIPTOMIC, 64, True),
    BaselineRow("Geneformer-backed clinical-association agent", Family.TRANSCRIPTOMIC, 65, True),
    BaselineRow("scGPT-backed clinical-association agent", Family.TRANSCRIPTOMIC, 66, True),
    BaselineRow(
        "TranscriptFormer-backed clinical-association agent", Family.TRANSCRIPTOMIC, 67, True
    ),
    BaselineRow(
        "Best-unconstrained ensemble (rank-averaged four-modality signal, no calibration, no provenance gate)",
        Family.FUSION_CONTROL,
        None,
        True,
    ),
    BaselineRow(
        "Single-monolithic-agent (unstructured evidence ingestion, one LLM call)",
        Family.FUSION_CONTROL,
        None,
        True,
    ),
)

DECLARED_BASELINES = 37
DECLARED_FUSION_CONTROLS = 2


class RegistryError(ValueError):
    """Raised when the transcribed table fails its own declared counts."""


def family_rows(family: Family) -> tuple[BaselineRow, ...]:
    return tuple(row for row in ROWS if row.family is family)


def family_counts() -> dict[str, int]:
    return {family.value: len(family_rows(family)) for family in Family}


def validate_registry() -> dict[str, object]:
    """Check the transcribed table against the counts its caption declares."""

    counts = family_counts()
    baseline_total = sum(
        count for family, count in counts.items() if family != Family.FUSION_CONTROL.value
    )
    control_total = counts[Family.FUSION_CONTROL.value]
    if baseline_total != DECLARED_BASELINES:
        raise RegistryError(
            f"family counts sum to {baseline_total}, the caption declares {DECLARED_BASELINES}"
        )
    if control_total != DECLARED_FUSION_CONTROLS:
        raise RegistryError(
            f"fusion controls number {control_total}, the caption declares {DECLARED_FUSION_CONTROLS}"
        )
    unknown = [row.name for row in ROWS if row.reference is not None and row.reference < 1]
    if unknown:
        raise RegistryError(f"invalid reference numbers for {unknown}")
    return {
        "baselines": baseline_total,
        "fusion_controls": control_total,
        "table_rows_including_proposed": baseline_total + control_total + 1,
        "family_counts": counts,
        "references": {row.name: row.reference for row in ROWS},
        "proposed_row": FUSION_PROPOSED,
    }


def all_row_names() -> tuple[str, ...]:
    return tuple(row.name for row in ROWS)


def research_rows() -> tuple[BaselineRow, ...]:
    return tuple(row for row in ROWS if row.reimplemented)


def original_benchmark_rows() -> tuple[BaselineRow, ...]:
    return tuple(row for row in ROWS if not row.reimplemented)
