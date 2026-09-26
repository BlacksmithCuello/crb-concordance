"""Genome-scale flux-feasibility agent.

Ref: Sec. 4.2 item (3) (a context-specific model is derived from the transcripts of
the tumour or cell line and the feasibility of alternative pathways under
suppression of g becomes m_g,FLUX; paired patient-derived transcriptome/cell-line
flux is the primary evidence and the purity-adjusted bulk flux score is a
lesser backup signal).
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.agents.base import AgentError, RawEvidence
from crb_concordance.metabolism.knockout import SuppressionAssessment, assess_gene_suppression
from crb_concordance.metabolism.model import StoichiometricModel
from crb_concordance.metrics.provenance import citation_key
from crb_concordance.utils.numerics import clip01
from crb_concordance.utils.types import Modality, ProvenanceRef

PRIMARY_WEIGHT = 0.7
BACKUP_WEIGHT = 0.3
PURITY_ATTENUATION = 0.6


@dataclass(frozen=True, slots=True)
class FluxAgentConfig:
    primary_weight: float = PRIMARY_WEIGHT
    backup_weight: float = BACKUP_WEIGHT
    purity_attenuation: float = PURITY_ATTENUATION

    def validate(self) -> None:
        total = self.primary_weight + self.backup_weight
        if abs(total - 1.0) > 1e-9:
            raise AgentError(f"primary and backup weights must sum to one: {total!r}")
        if not 0.0 <= self.purity_attenuation <= 1.0:
            raise AgentError(f"purity attenuation must lie in [0, 1]: {self.purity_attenuation!r}")


def purity_attenuated(score: float, *, attenuation: float = PURITY_ATTENUATION) -> float:
    """Discount a bulk-transcriptome flux score for the purity confound."""

    return clip01(score * attenuation)


class FluxFeasibilityAgent:
    """Measures how much flux feasibility the loss of a candidate removes."""

    name = "genome-scale flux feasibility"
    modality = Modality.FLUX

    def __init__(
        self,
        model: StoichiometricModel,
        *,
        backup_model: StoichiometricModel | None = None,
        config: FluxAgentConfig | None = None,
        source: str = "context-specific model extracted from Human-GEM v2.0.1",
        source_version: str = "Human-GEM v2.0.1, enzyme-constrained",
    ) -> None:
        self.model = model
        self.backup_model = backup_model
        self.config = config or FluxAgentConfig()
        self.config.validate()
        self.source = source
        self.source_version = source_version

    def assess(self, candidate: str) -> SuppressionAssessment:
        if not self.model.has_gene_rule(candidate):
            return SuppressionAssessment(
                gene=candidate,
                baseline_growth=0.0,
                suppressed_growth=0.0,
                blocked_reactions=(),
                pathway_feasibility=(),
            )
        return assess_gene_suppression(self.model, candidate)

    def collect(self, candidate: str) -> RawEvidence:
        assessment = self.assess(candidate)
        if not assessment.gated:
            return RawEvidence(
                candidate=candidate,
                modality=self.modality,
                score=None,
                evidence_nature="no reaction of the context-specific model is gated by the candidate",
                provenance=(
                    ProvenanceRef(
                        source=self.source,
                        version=self.source_version,
                        query=f"gpr={candidate}",
                    ),
                ),
                detail={"absent": True, "reason": "no gated reaction"},
            )
        primary = assessment.feasibility_score
        if self.backup_model is not None and self.backup_model.has_gene_rule(candidate):
            backup = assess_gene_suppression(self.backup_model, candidate).feasibility_score
            score = (
                self.config.primary_weight * primary
                + self.config.backup_weight
                * purity_attenuated(backup, attenuation=self.config.purity_attenuation)
            )
            nature = "paired cell-line flux feasibility with a purity-adjusted bulk backup"
        else:
            score = primary
            nature = "paired cell-line flux feasibility"
        return RawEvidence(
            candidate=candidate,
            modality=self.modality,
            score=clip01(score),
            evidence_nature=nature,
            provenance=(
                ProvenanceRef(
                    source=self.source,
                    version=self.source_version,
                    query=f"knockout={candidate}",
                ),
            ),
            citations=(
                citation_key(
                    candidate, self.name, self.source, self.source_version, f"knockout={candidate}"
                ),
            ),
            detail={
                "growth_feasibility_loss": assessment.growth_feasibility_loss,
                "pathway_capacity_loss": assessment.pathway_capacity_loss,
                "blocked_reactions": len(assessment.blocked_reactions),
                "pathways_assessed": len(assessment.pathway_feasibility),
            },
        )
