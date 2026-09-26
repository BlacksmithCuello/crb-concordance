"""The end-to-end discovery pass of Algorithm 1.

Ref: Sec. 4.2, Algorithm 1 (for each candidate collect the four evidences, discount
by Eq. (2), combine by Algorithm 2, append the per-agent entries and the
(Bel, Pl, BetP, K) to the provenance ledger, and return the pool sorted by BetP(V)
descending with intervals and traces attached).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crb_concordance.agents.base import AgentContext, RawEvidence
from crb_concordance.agents.dependency import DependencyAgent, DependencyAgentConfig
from crb_concordance.agents.flux_feasibility import FluxAgentConfig, FluxFeasibilityAgent
from crb_concordance.agents.kg_path import KGAgentConfig, KnowledgeGraphAgent
from crb_concordance.agents.mapping import MassMapper
from crb_concordance.agents.transcriptomic_clinical import (
    ClinicalAgentConfig,
    ClinicalAssociationAgent,
)
from crb_concordance.agents.verifier import VerificationOutcome, Verifier
from crb_concordance.belief.discount import DiscountRates
from crb_concordance.belief.trace import PLANNING_FLAG_THRESHOLD
from crb_concordance.discovery.ledger import ProvenanceLedger
from crb_concordance.discovery.ranking import RankedCandidate, rank_reports, score_map
from crb_concordance.metrics.provenance import (
    CitationAudit,
    audit_citations,
    ledger_index,
    summarise,
)
from crb_concordance.utils.types import MODALITY_ORDER, Modality

DEFAULT_EDGE_HOLDING = False


class DiscoveryError(ValueError):
    """Raised when a discovery pass is configured inconsistently."""


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    """Settings of one discovery pass."""

    conflict_threshold: float = PLANNING_FLAG_THRESHOLD
    max_hops: int = 3
    require_min_paths: int = 1
    purity_attenuation: float = 0.6
    clinical_min_observations: int = 12
    dependency_require_drug: bool = False
    combination_order: tuple[Modality, ...] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "conflict_threshold": self.conflict_threshold,
            "max_hops": self.max_hops,
            "require_min_paths": self.require_min_paths,
            "purity_attenuation": self.purity_attenuation,
            "clinical_min_observations": self.clinical_min_observations,
            "dependency_require_drug": self.dependency_require_drug,
            "combination_order": (
                [modality.value for modality in self.combination_order]
                if self.combination_order is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Ranked candidates with the ledger, audits and structural summary."""

    ranked: tuple[RankedCandidate, ...]
    ledger: ProvenanceLedger
    outcomes: tuple[VerificationOutcome, ...]
    audits: tuple[CitationAudit, ...]
    structural: dict[str, object]
    config: DiscoveryConfig

    def scores(self) -> dict[str, float]:
        return score_map(self.ranked)

    def provenance(self) -> dict[str, object]:
        return summarise(self.audits, tuple(self.ledger.entries)).as_dict()

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "ranking": [entry.as_row() for entry in self.ranked],
            "structural": self.structural,
            "provenance": self.provenance(),
        }


def build_agents(context: AgentContext, config: DiscoveryConfig) -> dict[Modality, object]:
    """Instantiate the four evidence producers plus the agents' declared names."""

    horizon = min(config.max_hops, context.finder.max_hops)
    return {
        Modality.KG: KnowledgeGraphAgent(
            context.finder,
            context.phenotype,
            config=KGAgentConfig(max_hops=horizon, min_paths=config.require_min_paths),
            blocked_edges=context.blocked_edges,
        ),
        Modality.DEP: DependencyAgent(
            context.dependency_table,
            config=DependencyAgentConfig(require_drug=config.dependency_require_drug),
        ),
        Modality.FLUX: FluxFeasibilityAgent(
            context.model,
            config=FluxAgentConfig(purity_attenuation=config.purity_attenuation),
        ),
        Modality.CLIN: ClinicalAssociationAgent(
            context.profiles,
            context.labels_by_record(),
            config=ClinicalAgentConfig(min_observations=config.clinical_min_observations),
        ),
    }


def collect_evidence(
    agents: dict[Modality, object], candidate: str
) -> tuple[dict[Modality, RawEvidence], dict[Modality, str]]:
    evidences: dict[Modality, RawEvidence] = {}
    names: dict[Modality, str] = {}
    for modality in MODALITY_ORDER:
        agent = agents[modality]
        evidences[modality] = agent.collect(candidate)  # type: ignore[attr-defined]
        names[modality] = str(agent.name)  # type: ignore[attr-defined]
    return evidences, names


def run_discovery(
    candidates: tuple[str, ...],
    context: AgentContext,
    rates: DiscountRates,
    *,
    config: DiscoveryConfig | None = None,
    mapper: MassMapper | None = None,
    ledger: ProvenanceLedger | None = None,
) -> DiscoveryResult:
    """Execute Algorithm 1 over the candidate pool."""

    settings = config or DiscoveryConfig()
    if not candidates:
        raise DiscoveryError("the candidate pool is empty")
    if settings.conflict_threshold < 0.0:
        raise DiscoveryError("the conflict-trace threshold must be non-negative")
    mass_mapper = mapper or MassMapper()
    verifier = Verifier(
        rates=rates, mapper=mass_mapper, conflict_threshold=settings.conflict_threshold
    )
    book = ledger if ledger is not None else ProvenanceLedger()
    agents = build_agents(context, settings)
    outcomes: list[VerificationOutcome] = []
    audits: list[CitationAudit] = []
    for candidate in candidates:
        evidences, names = collect_evidence(agents, candidate)
        outcome = verifier.combine_candidate(candidate, evidences, ledger=book, agent_names=names)
        outcomes.append(outcome)
        cited = tuple(
            citation for modality in MODALITY_ORDER for citation in evidences[modality].citations
        )
        if not cited:
            keys = book.keys()
            cited = tuple(keys[row] for row in outcome.ledger_rows)
        audits.append(audit_citations(candidate, cited, ledger_index(tuple(book.entries))))
    ranked = rank_reports(tuple(outcome.report for outcome in outcomes))
    return DiscoveryResult(
        ranked=ranked,
        ledger=book,
        outcomes=tuple(outcomes),
        audits=tuple(audits),
        structural=verifier.structural_report(tuple(outcomes)),
        config=settings,
    )


@dataclass(slots=True)
class DiscoveryCache:
    """Replays a discovery pass from a stored ledger without re-running the agents."""

    ledger: ProvenanceLedger
    stored: dict[str, dict[Modality, object]] = field(default_factory=dict)

    def load(self, rows: tuple[dict[str, object], ...]) -> None:
        for row in rows:
            candidate = str(row["candidate"])
            modality = Modality(str(row["modality"]))
            self.stored.setdefault(candidate, {})[modality] = row
