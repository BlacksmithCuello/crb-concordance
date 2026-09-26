"""Agent interface and shared context for the four evidence producers.

Ref: Sec. 4.1 and Sec. 4.2 (four agents indexed k in {KG, DEP, FLUX, CLIN}, each
transforming its raw evidence into a Dempster-Shafer mass function).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from crb_concordance.cohorts.control_panel import ControlPanel
from crb_concordance.cohorts.schema import ClinicalRecord, TranscriptomicRecord
from crb_concordance.graph.paths import PathFinder
from crb_concordance.metabolism.model import StoichiometricModel
from crb_concordance.utils.types import Availability, Modality, ProvenanceRef


class AgentError(ValueError):
    """Raised when an agent cannot produce evidence for a candidate."""


@dataclass(frozen=True, slots=True)
class RawEvidence:
    """One agent's raw evidence for one candidate, before the mass mapping."""

    candidate: str
    modality: Modality
    score: float | None
    evidence_nature: str
    provenance: tuple[ProvenanceRef, ...] = ()
    citations: tuple[str, ...] = ()
    detail: dict[str, float | str | bool | None] = field(default_factory=dict)

    @property
    def availability(self) -> Availability:
        return Availability.ABSENT if self.score is None else Availability.PRESENT

    @property
    def present(self) -> bool:
        return self.score is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "modality": self.modality.value,
            "score": self.score,
            "evidence_nature": self.evidence_nature,
            "availability": self.availability.value,
            "provenance": [ref.as_query_string() for ref in self.provenance],
            "citations": list(self.citations),
            "detail": dict(self.detail),
        }


class EvidenceAgent(Protocol):
    """Every evidence producer exposes its name, modality and a single collect call."""

    name: str
    modality: Modality

    def collect(self, candidate: str) -> RawEvidence: ...


@dataclass(slots=True)
class AgentContext:
    """Resources the agents read from, assembled once per run."""

    finder: PathFinder
    phenotype: str
    model: StoichiometricModel
    dependency_table: dict[str, tuple[float, float | None]]
    profiles: dict[str, TranscriptomicRecord]
    records: tuple[ClinicalRecord, ...]
    panel: ControlPanel
    medium: dict[str, float] = field(default_factory=dict)
    blocked_edges: frozenset[int] = frozenset()
    dependency_nature: str = "CRISPR Chronos gene effect and drug response"
    clinical_nature: str = "transcriptomic-clinical association"

    def labels_by_record(self) -> dict[str, int]:
        labelled: dict[str, int] = {}
        for record in self.records:
            if not record.retained():
                continue
            label = record.binary_label()
            if label is not None:
                labelled[record.record_id] = label
        return labelled

    def profile_matrix(self, gene: str) -> tuple[list[float], list[int]]:
        labelled = self.labels_by_record()
        values: list[float] = []
        targets: list[int] = []
        for record_id, label in labelled.items():
            profile = self.profiles.get(record_id)
            if profile is None:
                continue
            value = profile.gene(gene)
            if value is None:
                continue
            values.append(value)
            targets.append(label)
        return values, targets
