"""Shared vocabulary for the concordance pipeline.

Ref: Sec. 4.1 (frame Theta_g), Sec. 4.2 (provenance ledger fields).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


class Modality(str, Enum):
    """Evidence-producing modalities indexed k in the problem formulation."""

    KG = "KG"
    DEP = "DEP"
    FLUX = "FLUX"
    CLIN = "CLIN"


MODALITY_ORDER: tuple[Modality, ...] = (
    Modality.KG,
    Modality.DEP,
    Modality.FLUX,
    Modality.CLIN,
)


class Availability(str, Enum):
    """Whether a modality carried any measured evidence for a candidate."""

    PRESENT = "present"
    ABSENT = "absent"


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"


class LedgerEntry(TypedDict):
    """One provenance record produced by the critic/verifier agent."""

    candidate: str
    agent: str
    modality: str
    evidence_nature: str
    evidence_score: float | None
    mass_triple: tuple[float, float, float]
    discount_rate: float
    calibration_fold: int
    source: str
    source_version: str
    query: str
    recorded_at: str


class ClaimRecord(TypedDict):
    claim_id: str
    paper_location: str
    statement: str
    code_paths: list[str]
    verification: str


class DeviationRecord(TypedDict):
    paper_location: str
    departure: str
    justification: str


@dataclass(frozen=True, slots=True)
class ProvenanceRef:
    """A citable origin for one piece of evidence."""

    source: str
    version: str
    query: str
    release_date: str | None = None

    def as_query_string(self) -> str:
        parts = [self.source, self.version, self.query]
        if self.release_date is not None:
            parts.append(self.release_date)
        return "|".join(parts)
