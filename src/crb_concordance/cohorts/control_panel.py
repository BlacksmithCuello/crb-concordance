"""The retrospective control panel the discount rates are fitted on.

Ref: Sec. 4.1 (the discount rates are fitted by means of the retrospective control
panel built from the positive controls GLUT1/SLC2A1 and MCT1/SLC16A1 and the
negative controls SLC16A9, CPT1A and CES1, and are never recalibrated afterwards);
Sec. 4.4 (MCT1 is expected to score high while not deserving a high rank).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from crb_concordance.utils.numerics import rng_from
from crb_concordance.utils.types import MODALITY_ORDER, Modality


class ControlRole(str, Enum):
    POSITIVE = "positive"
    INFORMATIVE_NEGATIVE = "informative-negative"
    NEGATIVE = "negative"


class RankExpectation(str, Enum):
    HIGH = "high"
    DISCREPANT = "discrepant"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class ControlGene:
    """One control with its declared role and expected ranking behaviour."""

    symbol: str
    alias: str | None
    role: ControlRole
    expectation: RankExpectation
    rationale: str

    @property
    def label(self) -> int:
        return 1 if self.role is ControlRole.POSITIVE else 0

    @property
    def available_to_calibration(self) -> bool:
        return self.expectation is not RankExpectation.DISCREPANT


@dataclass(frozen=True, slots=True)
class ControlObservation:
    """Declared qualitative evidence pattern of one control across the four modalities."""

    symbol: str
    modality_scores: dict[Modality, float | None]

    def score(self, modality: Modality) -> float | None:
        return self.modality_scores.get(modality)


CONTROL_GENES: tuple[ControlGene, ...] = (
    ControlGene(
        symbol="SLC2A1",
        alias="GLUT1",
        role=ControlRole.POSITIVE,
        expectation=RankExpectation.HIGH,
        rationale="crossed the pharmacological, transcriptomic and clinical thresholds of the screen",
    ),
    ControlGene(
        symbol="SLC16A1",
        alias="MCT1",
        role=ControlRole.POSITIVE,
        expectation=RankExpectation.DISCREPANT,
        rationale="crossed the pharmacological threshold only, with no transcriptomic or cumulative radiosensitising trait",
    ),
    ControlGene(
        symbol="SLC16A9",
        alias="MCT9",
        role=ControlRole.NEGATIVE,
        expectation=RankExpectation.LOW,
        rationale="carnitine-uptake carrier used as a negative control of the screen",
    ),
    ControlGene(
        symbol="CPT1A",
        alias=None,
        role=ControlRole.NEGATIVE,
        expectation=RankExpectation.LOW,
        rationale="fatty-acid oxidation axis with the opposite direction of evidence",
    ),
    ControlGene(
        symbol="CES1",
        alias=None,
        role=ControlRole.NEGATIVE,
        expectation=RankExpectation.LOW,
        rationale="copy-number-linked predictor used as a negative control of the screen",
    ),
)

HOLD_OUT_POSITIVE = "SLC16A1"
CALIBRATION_GENES: tuple[str, ...] = tuple(
    gene.symbol for gene in CONTROL_GENES if gene.available_to_calibration
)


@dataclass(frozen=True, slots=True)
class ControlPanel:
    """The controls with their declared qualitative evidence pattern.

    The pattern itself is what the manuscript states: GLUT1 is positive on every
    axis, MCT1 is positive on the dependency axis alone, and the three negative
    controls are unsupported. The numeric scale that turns that pattern into mass
    functions is a declared engineering default, because the screen reports
    threshold crossings rather than the underlying scores.
    """

    genes: tuple[ControlGene, ...]
    observations: dict[str, ControlObservation]
    scale_note: str

    def labels(self) -> dict[str, int]:
        return {gene.symbol: gene.label for gene in self.genes}

    def ranks(self) -> dict[str, RankExpectation]:
        return {gene.symbol: gene.expectation for gene in self.genes}

    def observation(self, symbol: str) -> ControlObservation:
        try:
            return self.observations[symbol]
        except KeyError as error:
            raise KeyError(f"no control observation for {symbol}") from error

    def symbols(self) -> tuple[str, ...]:
        return tuple(gene.symbol for gene in self.genes)

    def positives(self) -> tuple[str, ...]:
        return tuple(gene.symbol for gene in self.genes if gene.role is ControlRole.POSITIVE)

    def negatives(self) -> tuple[str, ...]:
        return tuple(gene.symbol for gene in self.genes if gene.role is not ControlRole.POSITIVE)

    def calibration_symbols(self) -> tuple[str, ...]:
        return tuple(gene.symbol for gene in self.genes if gene.available_to_calibration)


SUPPORTED_SCORE = 0.82
UNSUPPORTED_SCORE = 0.12
DEFAULT_DISPERSION = 0.16
DEFAULT_PANEL_SEED = 20260101


def _coded(value: float, dispersion: float, generator: np.random.Generator) -> float:
    return float(min(max(value + generator.normal(0.0, dispersion), 0.0), 1.0))


def build_control_panel(
    *, dispersion: float = DEFAULT_DISPERSION, seed: int = DEFAULT_PANEL_SEED
) -> ControlPanel:
    """Encode the declared threshold crossings as modality evidence.

    The screen reports whether a control crossed each threshold rather than the
    underlying score, so the coded values carry a declared dispersion; without it the
    four controls would separate perfectly and the fitted rates would saturate at one.
    """

    generator = rng_from(seed)
    patterns = {
        "SLC2A1": (SUPPORTED_SCORE, SUPPORTED_SCORE, SUPPORTED_SCORE, SUPPORTED_SCORE),
        "SLC16A1": (SUPPORTED_SCORE, SUPPORTED_SCORE, UNSUPPORTED_SCORE, UNSUPPORTED_SCORE),
        "SLC16A9": (UNSUPPORTED_SCORE, UNSUPPORTED_SCORE, UNSUPPORTED_SCORE, UNSUPPORTED_SCORE),
        "CPT1A": (UNSUPPORTED_SCORE, UNSUPPORTED_SCORE, UNSUPPORTED_SCORE, UNSUPPORTED_SCORE),
        "CES1": (UNSUPPORTED_SCORE, UNSUPPORTED_SCORE, SUPPORTED_SCORE, UNSUPPORTED_SCORE),
    }
    observations = {
        symbol: ControlObservation(
            symbol=symbol,
            modality_scores={
                modality: _coded(pattern[index], dispersion, generator)
                for index, modality in enumerate(MODALITY_ORDER)
            },
        )
        for symbol, pattern in patterns.items()
    }
    return ControlPanel(
        genes=CONTROL_GENES,
        observations=observations,
        scale_note=(
            f"threshold crossings coded as {SUPPORTED_SCORE} (crossed) and "
            f"{UNSUPPORTED_SCORE} (not crossed) with declared dispersion {dispersion}; "
            "the screen reports crossings, not scores"
        ),
    )


def modality_observations(panel: ControlPanel, modality: Modality) -> tuple[tuple[str, float], ...]:
    """Control observations available to one modality, dropping absent entries."""

    collected: list[tuple[str, float]] = []
    for symbol in panel.symbols():
        value = panel.observation(symbol).score(modality)
        if value is None:
            continue
        collected.append((symbol, value))
    return tuple(collected)


def modality_labels(panel: ControlPanel, modality: Modality) -> tuple[int, ...]:
    observed = {symbol for symbol, _ in modality_observations(panel, modality)}
    return tuple(panel.labels()[symbol] for symbol in panel.symbols() if symbol in observed)


def declared_modalities() -> tuple[Modality, ...]:
    return MODALITY_ORDER
