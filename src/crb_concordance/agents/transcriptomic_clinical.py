"""Transcriptomic-clinical association agent.

Ref: Sec. 4.2 item (4) (expression of g is regressed against radiotherapy response
in the public and private cohorts and the magnitude of the association becomes
m_g,CLIN); Sec. 4.3 (the non-agentic multi-omics predictors and transcriptomic
signature papers are external comparators and never training data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.agents.base import AgentError, RawEvidence
from crb_concordance.cohorts.schema import TranscriptomicRecord
from crb_concordance.metrics.provenance import citation_key
from crb_concordance.utils.numerics import clip01
from crb_concordance.utils.types import Modality, ProvenanceRef

MIN_OBSERVATIONS = 12
ASSOCIATION_CEILING = 0.6


@dataclass(frozen=True, slots=True)
class ClinicalAgentConfig:
    min_observations: int = MIN_OBSERVATIONS
    association_ceiling: float = ASSOCIATION_CEILING
    private_cohort_weight: float = 0.5

    def validate(self) -> None:
        if self.min_observations < 3:
            raise AgentError(f"at least three observations are required: {self.min_observations!r}")
        if not 0.0 < self.association_ceiling <= 1.0:
            raise AgentError(
                f"association ceiling must lie in (0, 1]: {self.association_ceiling!r}"
            )
        if not 0.0 <= self.private_cohort_weight <= 1.0:
            raise AgentError(
                f"private cohort weight must lie in [0, 1]: {self.private_cohort_weight!r}"
            )


def point_biserial(values: np.ndarray, labels: np.ndarray) -> float:
    """Correlation between a continuous expression value and a binary response."""

    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.size != y.size:
        raise AgentError("expression and response vectors must align")
    if x.size < 3:
        raise AgentError("association needs at least three observations")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def association_strength(correlation: float, *, ceiling: float = ASSOCIATION_CEILING) -> float:
    return clip01(abs(correlation) / ceiling)


class ClinicalAssociationAgent:
    """Associates candidate expression with radiotherapy response."""

    name = "transcriptomic-clinical association"
    modality = Modality.CLIN

    def __init__(
        self,
        profiles: dict[str, TranscriptomicRecord],
        labels_by_record: dict[str, int],
        *,
        config: ClinicalAgentConfig | None = None,
        public_series: tuple[str, ...] = (
            "GSE35452",
            "GSE45404",
            "GSE68204",
            "GSE119409",
            "GSE150082",
        ),
        private_cohort_available: bool = False,
        source: str = "pretreatment-biopsy expression with pathological response",
        source_version: str = "GSE35452, GSE45404, GSE68204, GSE119409, GSE150082",
    ) -> None:
        self.profiles = profiles
        self.labels = labels_by_record
        self.config = config or ClinicalAgentConfig()
        self.config.validate()
        self.public_series = public_series
        self.private_cohort_available = private_cohort_available
        self.source = source
        self.source_version = source_version

    def paired_values(self, candidate: str) -> tuple[np.ndarray, np.ndarray]:
        values: list[float] = []
        targets: list[int] = []
        for record_id, label in self.labels.items():
            profile = self.profiles.get(record_id)
            if profile is None:
                continue
            value = profile.gene(candidate)
            if value is None:
                continue
            values.append(value)
            targets.append(label)
        return np.asarray(values, dtype=float), np.asarray(targets, dtype=float)

    def collect(self, candidate: str) -> RawEvidence:
        values, targets = self.paired_values(candidate)
        if values.size < self.config.min_observations:
            return RawEvidence(
                candidate=candidate,
                modality=self.modality,
                score=None,
                evidence_nature="too few paired expression and response observations",
                provenance=(
                    ProvenanceRef(
                        source=self.source, version=self.source_version, query=f"gene={candidate}"
                    ),
                ),
                detail={"absent": True, "observations": int(values.size)},
            )
        correlation = point_biserial(values, targets)
        score = association_strength(correlation, ceiling=self.config.association_ceiling)
        return RawEvidence(
            candidate=candidate,
            modality=self.modality,
            score=score,
            evidence_nature="transcriptomic-clinical association with radiotherapy response",
            provenance=(
                ProvenanceRef(
                    source=self.source,
                    version=self.source_version,
                    query=f"gene={candidate};series={len(self.public_series)}",
                ),
            ),
            citations=(
                citation_key(
                    candidate,
                    self.name,
                    self.source,
                    self.source_version,
                    f"gene={candidate};series={len(self.public_series)}",
                ),
            ),
            detail={
                "correlation": correlation,
                "observations": int(values.size),
                "private_cohort_available": self.private_cohort_available,
                "private_cohort_weight": self.config.private_cohort_weight,
            },
        )
