"""CRISPR/drug functional-dependency agent.

Ref: Sec. 4.2 item (2) (the agent queries functional-genomic dependency and
dose-response information on loss of fitness and drug sensitivity and turns it into
m_g,DEP); Sec. 4.5 (the dependency modality has no single comparator because gene
effect and drug response cannot be collapsed into one number).
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.agents.base import AgentError, RawEvidence
from crb_concordance.metrics.provenance import citation_key
from crb_concordance.utils.numerics import clip01
from crb_concordance.utils.types import Modality, ProvenanceRef

EFFECT_FLOOR = -0.1
EFFECT_SPAN = 1.2
DEPENDENCY_WEIGHT = 0.6
DRUG_WEIGHT = 0.4
DRUG_FLOOR = -0.2
DRUG_SPAN = 1.0


@dataclass(frozen=True, slots=True)
class DependencyAgentConfig:
    effect_floor: float = EFFECT_FLOOR
    effect_span: float = EFFECT_SPAN
    dependency_weight: float = DEPENDENCY_WEIGHT
    drug_weight: float = DRUG_WEIGHT
    require_drug: bool = False

    def validate(self) -> None:
        if self.effect_span <= 0.0:
            raise AgentError(f"effect span must be positive: {self.effect_span!r}")
        total = self.dependency_weight + self.drug_weight
        if abs(total - 1.0) > 1e-9:
            raise AgentError(f"dependency and drug weights must sum to one: {total!r}")


def dependency_score(
    effect: float, *, floor: float = EFFECT_FLOOR, span: float = EFFECT_SPAN
) -> float:
    """Map a Chronos gene effect to a dependency score: stronger loss is stronger support."""

    return clip01((floor - float(effect)) / span)


def drug_score(response: float, *, floor: float = DRUG_FLOOR, span: float = DRUG_SPAN) -> float:
    return clip01((floor - float(response)) / span)


class DependencyAgent:
    """Reads the functional-genomic and pharmacological evidence for a candidate."""

    name = "CRISPR/drug dependency"
    modality = Modality.DEP

    def __init__(
        self,
        dependency_table: dict[str, tuple[float, float | None]],
        *,
        config: DependencyAgentConfig | None = None,
        source: str = "DepMap CRISPR Chronos with GDSC, PRISM and LINCS L1000",
        source_version: str = "DepMap Public 26Q1, GDSC Release 8.4, CMap 2020 Level 5",
    ) -> None:
        self.table = dependency_table
        self.config = config or DependencyAgentConfig()
        self.config.validate()
        self.source = source
        self.source_version = source_version

    def collect(self, candidate: str) -> RawEvidence:
        entry = self.table.get(candidate)
        if entry is None:
            return self._absent(candidate, reason="candidate absent from the dependency release")
        effect, response = entry
        base = dependency_score(
            effect, floor=self.config.effect_floor, span=self.config.effect_span
        )
        if response is None:
            if self.config.require_drug:
                return self._absent(candidate, reason="no dose-response entry for the candidate")
            score = base
            nature = "functional-genomic loss of fitness"
        else:
            score = self.config.dependency_weight * base + self.config.drug_weight * drug_score(
                response
            )
            nature = "functional-genomic loss of fitness with dose-response sensitivity"
        return RawEvidence(
            candidate=candidate,
            modality=self.modality,
            score=score,
            evidence_nature=nature,
            provenance=(
                ProvenanceRef(
                    source=self.source,
                    version=self.source_version,
                    query=f"gene={candidate}",
                    release_date="2026-04-01",
                ),
            ),
            citations=(
                citation_key(
                    candidate, self.name, self.source, self.source_version, f"gene={candidate}"
                ),
            ),
            detail={
                "chronos_effect": effect,
                "drug_response": response,
                "dependency_component": base,
            },
        )

    def _absent(self, candidate: str, *, reason: str) -> RawEvidence:
        return RawEvidence(
            candidate=candidate,
            modality=self.modality,
            score=None,
            evidence_nature=reason,
            provenance=(
                ProvenanceRef(
                    source=self.source,
                    version=self.source_version,
                    query=f"gene={candidate}",
                ),
            ),
            detail={"absent": True, "reason": reason},
        )
