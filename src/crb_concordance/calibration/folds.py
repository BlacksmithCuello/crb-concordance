"""Calibration folds over the control panel.

Ref: Sec. 4.1 (the retrospective calibration fold is built like the four-fold panel;
every fold uses three of the controls together with the whole set of negative
controls, and one positive is never processed, so the Recall@k contribution of the
withheld positive is computed from rates calibrated without it).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from crb_concordance.cohorts.control_panel import ControlPanel


class FoldPolicy(str, Enum):
    LEAVE_ONE_CONTROL_OUT = "leave_one_control_out"
    FULL_PANEL = "full_panel"


class FoldError(ValueError):
    """Raised when a fold cannot be formed from the declared controls."""


@dataclass(frozen=True, slots=True)
class CalibrationFold:
    """One fold: the controls used for fitting and the control withheld from it."""

    index: int
    training_symbols: tuple[str, ...]
    held_out_symbol: str | None

    @property
    def is_held_out(self) -> bool:
        return self.held_out_symbol is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "training_symbols": list(self.training_symbols),
            "held_out_symbol": self.held_out_symbol,
        }


def build_folds(
    panel: ControlPanel, policy: FoldPolicy = FoldPolicy.LEAVE_ONE_CONTROL_OUT
) -> tuple[CalibrationFold, ...]:
    """Four leave-one-out folds over the calibration controls, or a single full-panel fold."""

    symbols = panel.calibration_symbols()
    if policy is FoldPolicy.FULL_PANEL:
        return (CalibrationFold(index=0, training_symbols=symbols, held_out_symbol=None),)
    if len(symbols) < 2:
        raise FoldError("leave-one-out calibration needs at least two controls")
    folds: list[CalibrationFold] = []
    for index, held_out in enumerate(symbols):
        training = tuple(symbol for symbol in symbols if symbol != held_out)
        folds.append(
            CalibrationFold(index=index, training_symbols=training, held_out_symbol=held_out)
        )
    return tuple(folds)


def fold_for_symbol(folds: tuple[CalibrationFold, ...], symbol: str) -> CalibrationFold:
    for fold in folds:
        if fold.held_out_symbol == symbol:
            return fold
    raise FoldError(f"no fold withholds {symbol}")


def withheld_positives(panel: ControlPanel, folds: tuple[CalibrationFold, ...]) -> tuple[str, ...]:
    """Positive controls excluded from the fold that reports their own Recall@k."""

    positives = set(panel.positives())
    return tuple(
        fold.held_out_symbol
        for fold in folds
        if fold.held_out_symbol is not None and fold.held_out_symbol in positives
    )


def audit_folds(panel: ControlPanel, folds: tuple[CalibrationFold, ...]) -> dict[str, object]:
    covered = {fold.held_out_symbol for fold in folds}
    return {
        "policy_size": len(folds),
        "calibration_symbols": list(panel.calibration_symbols()),
        "held_out_symbols": sorted(symbol for symbol in covered if symbol is not None),
        "every_symbol_withheld_once": covered == set(panel.calibration_symbols()),
        "withheld_positives": list(withheld_positives(panel, folds)),
        "never_calibrated": [
            gene.symbol for gene in panel.genes if not gene.available_to_calibration
        ],
    }
