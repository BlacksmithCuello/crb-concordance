"""Knowledge-graph path retrieval and grounding agent.

Ref: Sec. 4.2 item (1) (multi-hop routes between the target and phenotype nodes are
retrieved on the metabolic-reaction-extended substrate and the support measured at
the pathway level is converted into m_g,KG); Sec. 4.4 (the substrate-circularity
and edge-holding controls apply to exactly this agent).
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.agents.base import AgentError, RawEvidence
from crb_concordance.graph.paths import PathEvidence, PathFinder, collect_path_evidence
from crb_concordance.metrics.provenance import citation_key
from crb_concordance.utils.types import Modality, ProvenanceRef


@dataclass(frozen=True, slots=True)
class KGAgentConfig:
    max_hops: int = 3
    min_paths: int = 1

    def validate(self) -> None:
        if self.max_hops < 1:
            raise AgentError(f"max_hops must be positive: {self.max_hops!r}")
        if self.min_paths < 1:
            raise AgentError(f"min_paths must be positive: {self.min_paths!r}")


class KnowledgeGraphAgent:
    """Grounds a candidate by the pathway-level support of its retrieved routes."""

    name = "knowledge-graph path retrieval"
    modality = Modality.KG

    def __init__(
        self,
        finder: PathFinder,
        phenotype: str,
        *,
        config: KGAgentConfig | None = None,
        blocked_edges: frozenset[int] = frozenset(),
        source_version: str = "OptimusKG extended with Human-GEM v2.0.1",
        source: str = "curated biomedical substrate",
    ) -> None:
        self.config = config or KGAgentConfig()
        self.config.validate()
        if self.config.max_hops > finder.max_hops:
            raise AgentError(
                f"agent horizon {self.config.max_hops} exceeds the finder bound {finder.max_hops}"
            )
        self.finder = finder
        self.phenotype = phenotype
        self.blocked_edges = blocked_edges
        self.source = source
        self.source_version = source_version

    def path_evidence(self, candidate: str) -> PathEvidence:
        return collect_path_evidence(
            self.finder, candidate, self.phenotype, blocked_edges=self.blocked_edges
        )

    def collect(self, candidate: str) -> RawEvidence:
        evidence = self.path_evidence(candidate)
        score = evidence.support if len(evidence.paths) >= self.config.min_paths else None
        provenance = tuple(
            sorted({edge.source_db for path in evidence.paths for edge in path.edges})
        )
        references = tuple(self._reference(database) for database in provenance) or (
            self._reference("none"),
        )
        primary = references[0]
        return RawEvidence(
            candidate=evidence.candidate,
            modality=self.modality,
            score=score,
            evidence_nature="graph-topology association test",
            provenance=references,
            citations=(
                citation_key(
                    evidence.candidate, self.name, primary.source, primary.version, primary.query
                ),
            ),
            detail={
                "paths": len(evidence.paths),
                "edge_references": list(evidence.citations()),
                "databases": list(provenance),
                "min_hops": min(evidence.hops) if evidence.hops else 0,
                "max_hops": max(evidence.hops) if evidence.hops else 0,
                "pathways": len(evidence.pathway_hits),
                "direct_edge_present": evidence.direct_edge_present,
                "edge_holding_active": bool(self.blocked_edges),
            },
        )

    def _reference(self, database: str) -> ProvenanceRef:
        return ProvenanceRef(
            source=self.source,
            version=self.source_version,
            query=f"hops<={self.config.max_hops};db={database}",
            release_date=database,
        )
