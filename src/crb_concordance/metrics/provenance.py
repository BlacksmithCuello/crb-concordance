"""Provenance-grounding and hallucinated-edge rates.

Ref: Sec. 4.4 (the provenance-grounding rate is the fraction of a candidate's cited
evidence tracing to a verifiable ledger entry; the hallucinated-edge rate is the
complementary fraction, audited rather than self-reported); Sec. 4.2 (the ledger
records agent, candidate, evidence nature, score, mass triple, discount rate,
calibration fold, source, version, query and timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.utils.types import LedgerEntry


class ProvenanceError(ValueError):
    """Raised when a citation audit is asked for malformed ledger data."""


@dataclass(frozen=True, slots=True)
class CitationAudit:
    """One candidate's citation audit against the ledger."""

    candidate: str
    cited: int
    grounded: int

    @property
    def grounded_rate(self) -> float:
        if self.cited == 0:
            return 0.0
        return self.grounded / self.cited

    @property
    def hallucinated_rate(self) -> float:
        return 1.0 - self.grounded_rate

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "candidate": self.candidate,
            "cited": self.cited,
            "grounded": self.grounded,
            "provenance_grounding_rate": self.grounded_rate,
            "hallucinated_edge_rate": self.hallucinated_rate,
        }


def ledger_key(entry: LedgerEntry) -> str:
    return citation_key(
        entry["candidate"], entry["agent"], entry["source"], entry["source_version"], entry["query"]
    )


def citation_key(candidate: str, agent: str, source: str, source_version: str, query: str) -> str:
    """The identifier an agent asserts for one of its calls, and the ledger's own key."""

    return f"{candidate}|{agent}|{source}|{source_version}|{query}"


def ledger_index(entries: tuple[LedgerEntry, ...]) -> dict[str, LedgerEntry]:
    return {ledger_key(entry): entry for entry in entries}


def audit_citations(
    candidate: str,
    citations: tuple[str, ...],
    index: dict[str, LedgerEntry],
) -> CitationAudit:
    if not citations:
        raise ProvenanceError(f"candidate {candidate} cites no evidence")
    grounded = sum(1 for citation in citations if citation in index)
    return CitationAudit(candidate=candidate, cited=len(citations), grounded=grounded)


def provenance_grounding_rate(audits: tuple[CitationAudit, ...]) -> float:
    cited = sum(audit.cited for audit in audits)
    if cited == 0:
        return 0.0
    return sum(audit.grounded for audit in audits) / cited


def hallucinated_edge_rate(audits: tuple[CitationAudit, ...]) -> float:
    return 1.0 - provenance_grounding_rate(audits)


@dataclass(frozen=True, slots=True)
class ProvenanceSummary:
    candidates: int
    cited: int
    grounded: int
    grounded_rate: float
    hallucinated_rate: float
    complete_entries: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "candidates": self.candidates,
            "cited": self.cited,
            "grounded": self.grounded,
            "provenance_grounding_rate": self.grounded_rate,
            "hallucinated_edge_rate": self.hallucinated_rate,
            "complete_entries": self.complete_entries,
        }


NULLABLE_LEDGER_FIELDS: frozenset[str] = frozenset({"evidence_score"})

REQUIRED_LEDGER_FIELDS: tuple[str, ...] = (
    "candidate",
    "agent",
    "modality",
    "evidence_nature",
    "evidence_score",
    "mass_triple",
    "discount_rate",
    "calibration_fold",
    "source",
    "source_version",
    "query",
    "recorded_at",
)


def complete_entries(entries: tuple[LedgerEntry, ...]) -> int:
    """Ledger rows carrying every field the system architecture declares."""

    count = 0
    for entry in entries:
        row = dict(entry)
        complete = all(field in row for field in REQUIRED_LEDGER_FIELDS)
        complete = complete and all(
            row[field] is not None
            for field in REQUIRED_LEDGER_FIELDS
            if field not in NULLABLE_LEDGER_FIELDS
        )
        if complete:
            count += 1
    return count


def summarise(
    audits: tuple[CitationAudit, ...], entries: tuple[LedgerEntry, ...]
) -> ProvenanceSummary:
    cited = sum(audit.cited for audit in audits)
    grounded = sum(audit.grounded for audit in audits)
    rate = grounded / cited if cited else 0.0
    return ProvenanceSummary(
        candidates=len(audits),
        cited=cited,
        grounded=grounded,
        grounded_rate=rate,
        hallucinated_rate=1.0 - rate,
        complete_entries=complete_entries(entries),
    )


def unsupported_ledger_fields(entries: tuple[LedgerEntry, ...]) -> dict[str, int]:
    """How many ledger rows leave each declared field empty."""

    tally: dict[str, int] = {}
    rows = [dict(entry) for entry in entries]
    for field in REQUIRED_LEDGER_FIELDS:
        if field in NULLABLE_LEDGER_FIELDS:
            missing = sum(1 for row in rows if field not in row)
        else:
            missing = sum(1 for row in rows if field not in row or row[field] is None)
        if missing:
            tally[field] = missing
    return tally
