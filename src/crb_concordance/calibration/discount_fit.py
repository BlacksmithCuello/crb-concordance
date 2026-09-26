"""Fitting the per-modality discount rates on the retrospective control panel.

Ref: Sec. 4.1, Eq. (2) and the paragraph fitting {alpha_k}; Sec. 4.2, Algorithm 3
(discovery time reuses the fixed, already-calibrated rates with no parameter
update, so the fit is a closed-form deterministic mixture).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from crb_concordance.belief.discount import DiscountRates
from crb_concordance.calibration.folds import CalibrationFold, FoldPolicy, build_folds
from crb_concordance.cohorts.control_panel import ControlPanel
from crb_concordance.metrics.calibration import brier_score
from crb_concordance.utils.numerics import clip01, golden_section_minimum
from crb_concordance.utils.types import MODALITY_ORDER, Modality

RATE_BOUNDS = (0.0, 1.0)


class CalibrationError(ValueError):
    """Raised when a discount rate cannot be fitted from the supplied panel."""


@dataclass(frozen=True, slots=True)
class ModalityFit:
    """One modality's fitted rate with the panel loss it attains."""

    modality: Modality
    rate: float
    loss: float
    brier_at_unit_rate: float
    observations: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "modality": self.modality.value,
            "rate": self.rate,
            "brier": self.loss,
            "brier_at_unit_rate": self.brier_at_unit_rate,
            "observations": self.observations,
        }


@dataclass(frozen=True, slots=True)
class FoldFit:
    """The four fitted rates of one calibration fold."""

    fold: CalibrationFold
    fits: tuple[ModalityFit, ...]

    def rates(self) -> DiscountRates:
        return DiscountRates({fit.modality: fit.rate for fit in self.fits})

    def as_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold.as_dict(),
            "fits": [fit.as_dict() for fit in self.fits],
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Every fold fit plus the pooled rates used for discovery-time scoring."""

    folds: tuple[FoldFit, ...]
    pooled: DiscountRates
    policy: FoldPolicy
    details: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.value,
            "pooled_rates": self.pooled.as_dict(),
            "folds": [fold.as_dict() for fold in self.folds],
            **self.details,
        }


def single_modality_pignistic(score: float, rate: float) -> float:
    """BetP(V) of one modality whose raw committed mass equals its evidence score."""

    return rate * score + 0.5 * (1.0 - rate)


def rate_loss(scores: np.ndarray, labels: np.ndarray, rate: float) -> float:
    predicted = np.asarray([single_modality_pignistic(float(s), rate) for s in scores], dtype=float)
    return brier_score(predicted, labels)


def fit_rate_closed_form(scores: np.ndarray, labels: np.ndarray) -> float:
    """Minimiser of the quadratic panel loss, the closed form of Eq. (2) fitting."""

    a = np.asarray(scores, dtype=float) - 0.5
    b = 0.5 - np.asarray(labels, dtype=float)
    denominator = float(np.sum(a * a))
    if denominator <= 1e-15:
        return 1.0
    return clip01(float(-np.sum(a * b) / denominator))


def fit_rate_numeric(scores: np.ndarray, labels: np.ndarray) -> float:
    """Derivative-free minimiser of the same loss, used to check the closed form."""

    candidate, _ = golden_section_minimum(
        lambda rate: rate_loss(scores, labels, rate), *RATE_BOUNDS, tolerance=1e-12
    )
    return clip01(candidate)


def fit_modality(
    panel: ControlPanel,
    modality: Modality,
    symbols: tuple[str, ...] | None,
    *,
    numeric: bool = False,
) -> ModalityFit:
    """Fit one modality's rate on the controls listed in ``symbols``."""

    labels = panel.labels()
    selected = symbols if symbols is not None else panel.symbols()
    scores: list[float] = []
    targets: list[float] = []
    for symbol in selected:
        value = panel.observation(symbol).score(modality)
        if value is None:
            continue
        scores.append(float(value))
        targets.append(float(labels[symbol]))
    if not scores:
        raise CalibrationError(f"no control observations available for {modality.value}")
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(targets, dtype=float)
    if numeric:
        rate = fit_rate_numeric(score_array, label_array)
    else:
        interior = fit_rate_closed_form(score_array, label_array)
        candidates = (interior, RATE_BOUNDS[0], RATE_BOUNDS[1])
        rate = min(candidates, key=lambda value: rate_loss(score_array, label_array, value))
    return ModalityFit(
        modality=modality,
        rate=rate,
        loss=rate_loss(score_array, label_array, rate),
        brier_at_unit_rate=rate_loss(score_array, label_array, 1.0),
        observations=len(scores),
    )


def fit_fold(panel: ControlPanel, fold: CalibrationFold, *, numeric: bool = False) -> FoldFit:
    fits = tuple(
        fit_modality(panel, modality, fold.training_symbols, numeric=numeric)
        for modality in MODALITY_ORDER
    )
    return FoldFit(fold=fold, fits=fits)


def calibrate(
    panel: ControlPanel,
    *,
    policy: FoldPolicy = FoldPolicy.LEAVE_ONE_CONTROL_OUT,
    numeric: bool = False,
) -> CalibrationReport:
    """Fit every fold and pool the rates the discovery pass uses."""

    folds = build_folds(panel, policy)
    fitted = tuple(fit_fold(panel, fold, numeric=numeric) for fold in folds)
    pooled_fits = tuple(
        fit_modality(panel, modality, panel.calibration_symbols(), numeric=numeric)
        for modality in MODALITY_ORDER
    )
    pooled = DiscountRates({fit.modality: fit.rate for fit in pooled_fits})
    held_out_recall_provenance = {
        fold.fold.held_out_symbol: {
            "calibration_fold": fold.fold.index,
            "rates": fold.rates().as_dict(),
        }
        for fold in fitted
        if fold.fold.held_out_symbol is not None
    }
    return CalibrationReport(
        folds=fitted,
        pooled=pooled,
        policy=policy,
        details={
            "pooled_fits": [fit.as_dict() for fit in pooled_fits],
            "ignorance_floor": pooled.ignorance_floor(),
            "reliability_order": [modality.value for modality in pooled.reliability_order()],
            "held_out_rate_provenance": held_out_recall_provenance,
        },
    )


def rate_stability(report: CalibrationReport) -> dict[str, float]:
    """Largest spread of a rate across folds, the fold-to-fold stability check."""

    spread: dict[str, float] = {}
    for modality in MODALITY_ORDER:
        values = [
            next(fit.rate for fit in fold.fits if fit.modality is modality) for fold in report.folds
        ]
        spread[modality.value] = float(max(values) - min(values))
    return spread


def closed_form_agreement(scores: np.ndarray, labels: np.ndarray) -> float:
    """Absolute gap between the closed-form and derivative-free rate."""

    return abs(fit_rate_closed_form(scores, labels) - fit_rate_numeric(scores, labels))
