"""The fixed calibrated map from raw evidence to a mass triple.

Ref: Sec. 4.2, Algorithm 3 step 2 (m_g,k is the fixed, already-fit mapping
E_k(g) -> (m(V), m(not V), m(Theta))) and step 3 (the discounting of Eq. (2));
Sec. 4.1 (absent evidence must stay distinct from contradictory evidence, so an
absent modality contributes the vacuous mass and never a committed one).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.agents.base import RawEvidence
from crb_concordance.belief.mass import MassTriple, vacuous
from crb_concordance.utils.numerics import clip01, sigmoid
from crb_concordance.utils.types import MODALITY_ORDER, Modality


class MappingError(ValueError):
    """Raised when a mapping parameter falls outside its declared range."""


@dataclass(frozen=True, slots=True)
class MassMappingParams:
    """Per-modality slope, offset and commitment gate of the evidence-to-mass map."""

    slope: float
    offset: float
    commitment: float

    def validate(self) -> None:
        if not 0.0 < self.commitment <= 1.0:
            raise MappingError(f"commitment must lie in (0, 1]: {self.commitment!r}")

    def posterior(self, score: float) -> float:
        return float(sigmoid(self.slope * clip01(score) + self.offset))

    def as_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "offset": self.offset, "commitment": self.commitment}


DEFAULT_SLOPE = 6.0
DEFAULT_OFFSET = -3.0
DEFAULT_COMMITMENT = 0.9
MIN_COMMITTED_MASS = 1e-6


def default_params() -> dict[Modality, MassMappingParams]:
    """A deterministic commitment schedule, one gate per modality.

    The manuscript fixes the mapping as an already-fit object without reporting its
    values, so the shipped defaults are an engineering choice; the per-modality
    gates follow the reliability order the paper uses for the combination sequence.
    """

    gates = {
        Modality.KG: 0.88,
        Modality.DEP: 0.92,
        Modality.FLUX: 0.80,
        Modality.CLIN: 0.72,
    }
    return {
        modality: MassMappingParams(
            slope=DEFAULT_SLOPE, offset=DEFAULT_OFFSET, commitment=gates[modality]
        )
        for modality in MODALITY_ORDER
    }


class MassMapper:
    """Turns raw evidence into a mass triple, keeping absence genuinely vacuous."""

    def __init__(self, params: dict[Modality, MassMappingParams] | None = None) -> None:
        chosen = params if params is not None else default_params()
        missing = [modality for modality in MODALITY_ORDER if modality not in chosen]
        if missing:
            raise MappingError(f"missing mapping parameters for {[m.value for m in missing]}")
        for value in chosen.values():
            value.validate()
        self._params = dict(chosen)

    def params(self, modality: Modality) -> MassMappingParams:
        return self._params[modality]

    def map_score(self, modality: Modality, score: float) -> MassTriple:
        parameters = self._params[modality]
        posterior = parameters.posterior(score)
        committed = max(parameters.commitment, MIN_COMMITTED_MASS)
        return MassTriple(
            v=committed * posterior,
            n=committed * (1.0 - posterior),
            theta=1.0 - committed,
        )

    def map_evidence(self, evidence: RawEvidence) -> MassTriple:
        if evidence.score is None:
            return vacuous()
        return self.map_score(evidence.modality, evidence.score)

    def map_scores(self, scores: dict[Modality, float | None]) -> dict[Modality, MassTriple]:
        return {
            modality: (
                vacuous()
                if scores.get(modality) is None
                else self.map_score(modality, float(scores[modality]))
            )
            for modality in MODALITY_ORDER
        }

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {modality.value: self._params[modality].as_dict() for modality in MODALITY_ORDER}


def posterior_vector(scores: np.ndarray, params: MassMappingParams) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(-(params.slope * clipped + params.offset)))


def mass_tensor(scores: np.ndarray, params: MassMappingParams) -> np.ndarray:
    """Stacked (v, n, theta) rows for a batch of scores."""

    posterior = posterior_vector(scores, params)
    committed = np.full_like(posterior, params.commitment)
    return np.column_stack([committed * posterior, committed * (1.0 - posterior), 1.0 - committed])


def absent_mask(scores: np.ndarray) -> np.ndarray:
    return np.isnan(np.asarray(scores, dtype=float))
