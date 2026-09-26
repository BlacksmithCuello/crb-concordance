"""Append-only provenance ledger.

Ref: Sec. 4.2 (the ledger records the agent and applicant of each call, the nature
of the evidence, the score, the (m(V), m(not V), m(Theta)) triple, the discount rate
and calibration fold, the source, version and query, and the date and time, and can
only be extended by appending).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from crb_concordance.agents.base import RawEvidence
from crb_concordance.belief.mass import AgentMass
from crb_concordance.utils.atomic import append_jsonl
from crb_concordance.utils.types import LedgerEntry, Modality


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class LedgerError(ValueError):
    """Raised when a ledger row would be written without its declared fields."""


@dataclass(slots=True)
class ProvenanceLedger:
    """In-memory ledger that serialises to JSON lines."""

    clock: Callable[[], str] = utc_now
    entries: list[LedgerEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def record(
        self,
        *,
        candidate: str,
        agent: str,
        modality: Modality,
        evidence: RawEvidence,
        mass: AgentMass,
    ) -> LedgerEntry:
        reference = evidence.provenance[0] if evidence.provenance else None
        row: LedgerEntry = {
            "candidate": candidate,
            "agent": agent,
            "modality": modality.value,
            "evidence_nature": evidence.evidence_nature,
            "evidence_score": evidence.score,
            "mass_triple": mass.triple.as_tuple(),
            "discount_rate": mass.discount_rate,
            "calibration_fold": mass.calibration_fold,
            "source": reference.source if reference is not None else "",
            "source_version": reference.version if reference is not None else "",
            "query": reference.query if reference is not None else "",
            "recorded_at": self.clock(),
        }
        self.entries.append(row)
        return row

    def record_all(self, rows: list[LedgerEntry]) -> None:
        self.entries.extend(rows)

    def keys(self) -> tuple[str, ...]:
        return tuple(
            f"{row['candidate']}|{row['agent']}|{row['source']}|{row['source_version']}|{row['query']}"
            for row in self.entries
        )

    def flush(self, path: str | Path) -> Path:
        if not self.entries:
            raise LedgerError("refusing to flush an empty ledger")
        return append_jsonl(path, [dict(row) for row in self.entries])

    def per_candidate(self, candidate: str) -> tuple[LedgerEntry, ...]:
        return tuple(row for row in self.entries if row["candidate"] == candidate)

    def as_dicts(self) -> list[dict[str, object]]:
        return [dict(row) for row in self.entries]
