"""Differentiable calibration of the evidence-to-mass map.

Ref: Sec. 4.2, Algorithm 3 step 2 (the mapping E_k(g) -> (m(V), m(not V), m(Theta))
is fitted once at calibration time and then fixed); Sec. 4.1 (the pre-discount mass
function serves as an estimator of the posterior for the modality considered);
Sec. 4.4 (ECE and Brier score judge whether BetP(V) behaves as a calibrated
probability against the control panel's known labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from crb_concordance.agents.mapping import MassMapper, MassMappingParams
from crb_concordance.cohorts.control_panel import ControlPanel
from crb_concordance.metrics.calibration import brier_score, expected_calibration_error
from crb_concordance.training.engine import TrainingHistory, fit
from crb_concordance.training.optim import OptimConfig
from crb_concordance.utils.types import MODALITY_ORDER, Modality

GATE_FLOOR = 1e-4


class MassCalibrationError(ValueError):
    """Raised when the calibration table cannot be assembled."""


class CalibratedMassMap(torch.nn.Module):
    """Per-modality slope, offset and commitment gate of the mass map."""

    def __init__(
        self,
        *,
        n_modalities: int = len(MODALITY_ORDER),
        slope: float = 6.0,
        offset: float = -3.0,
        gate: float = 0.9,
    ) -> None:
        super().__init__()
        if n_modalities < 1:
            raise MassCalibrationError("at least one modality is required")
        self.slope = torch.nn.Parameter(torch.full((n_modalities,), float(slope)))
        self.offset = torch.nn.Parameter(torch.full((n_modalities,), float(offset)))
        self.gate_logit = torch.nn.Parameter(
            torch.full((n_modalities,), float(np.log(gate / (1.0 - gate))))
        )

    def posterior(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.slope * scores + self.offset)

    def commitment(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        posterior = self.posterior(scores)
        gate = self.commitment()
        v = gate * posterior
        n = gate * (1.0 - posterior)
        theta = (1.0 - gate).expand_as(posterior)
        return torch.stack([v, n, theta], dim=-1)

    def pignistic(self, scores: torch.Tensor) -> torch.Tensor:
        masses = self.forward(scores)
        return masses[..., 0] + 0.5 * masses[..., 2]

    def to_params(self) -> dict[Modality, MassMappingParams]:
        gate = self.commitment().detach().cpu().numpy()
        slope = self.slope.detach().cpu().numpy()
        offset = self.offset.detach().cpu().numpy()
        return {
            modality: MassMappingParams(
                slope=float(slope[index]),
                offset=float(offset[index]),
                commitment=float(np.clip(gate[index], GATE_FLOOR, 1.0)),
            )
            for index, modality in enumerate(MODALITY_ORDER)
        }


@dataclass(frozen=True, slots=True)
class CalibrationTable:
    """Scores, labels and availability masks, one row per control observation."""

    scores: np.ndarray
    labels: np.ndarray
    mask: np.ndarray

    @property
    def observations(self) -> int:
        return int(self.scores.shape[0])

    def as_dict(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "present": int(np.count_nonzero(self.mask)),
            "modalities": [modality.value for modality in MODALITY_ORDER],
        }


def build_table(panel: ControlPanel) -> CalibrationTable:
    """One row per control, one column per modality, absent observations masked."""

    symbols = panel.symbols()
    if not symbols:
        raise MassCalibrationError("the control panel is empty")
    scores = np.zeros((len(symbols), len(MODALITY_ORDER)), dtype=float)
    mask = np.zeros_like(scores, dtype=bool)
    for row, symbol in enumerate(symbols):
        observation = panel.observation(symbol)
        for column, modality in enumerate(MODALITY_ORDER):
            value = observation.score(modality)
            if value is None:
                continue
            scores[row, column] = float(value)
            mask[row, column] = True
    labels = np.asarray([float(panel.labels()[symbol]) for symbol in symbols], dtype=float)[:, None]
    if not np.any(mask):
        raise MassCalibrationError("no control observation carries a score")
    return CalibrationTable(scores=scores, labels=labels, mask=mask)


def batches_from_table(table: CalibrationTable, batch_size: int) -> list[tuple[torch.Tensor, ...]]:
    if batch_size < 1:
        raise MassCalibrationError(f"batch size must be positive: {batch_size!r}")
    scores = torch.tensor(table.scores, dtype=torch.float32)
    labels = torch.tensor(table.labels, dtype=torch.float32)
    mask = torch.tensor(table.mask, dtype=torch.bool)
    batches: list[tuple[torch.Tensor, ...]] = []
    for start in range(0, table.observations, batch_size):
        stop = min(start + batch_size, table.observations)
        batches.append((scores[start:stop], labels[start:stop], mask[start:stop]))
    return batches


def brier_mass_loss(
    module: CalibratedMassMap, scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Brier loss of the pignistic estimate over the available modality entries."""

    predictions = module.pignistic(scores)
    errors = (predictions - labels) ** 2
    selected = errors[mask]
    if selected.numel() == 0:
        return module.offset.sum() * 0.0
    return selected.mean()


@dataclass(frozen=True, slots=True)
class MassCalibrationReport:
    """The fitted mapper with the panel losses it attains."""

    mapper: MassMapper
    before: dict[str, float]
    after: dict[str, float]
    history: TrainingHistory

    def as_dict(self) -> dict[str, object]:
        return {
            "before": self.before,
            "after": self.after,
            "parameters": self.mapper.as_dict(),
            "training": self.history.as_dict(),
        }


def _panel_metrics(mapper: MassMapper, table: CalibrationTable) -> dict[str, float]:
    predictions: list[float] = []
    targets: list[float] = []
    for row in range(table.observations):
        label = float(table.labels[row, 0])
        values: list[float] = []
        for column, modality in enumerate(MODALITY_ORDER):
            if not table.mask[row, column]:
                continue
            triple = mapper.map_score(modality, float(table.scores[row, column]))
            values.append(triple.pignistic())
        predictions.append(float(np.mean(values)))
        targets.append(label)
    predicted = np.asarray(predictions, dtype=float)
    observed = np.asarray(targets, dtype=float)
    return {
        "brier": brier_score(predicted, observed),
        "expected_calibration_error": expected_calibration_error(predicted, observed),
        "mean_probability": float(np.mean(predicted)),
        "base_rate": float(np.mean(observed)),
    }


def fit_mass_map(
    panel: ControlPanel,
    *,
    config: OptimConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> MassCalibrationReport:
    """Fit the mass map on the control panel and return it as a plain mapper."""

    table = build_table(panel)
    settings = config or OptimConfig()
    settings.validate()
    module = CalibratedMassMap()
    initial = MassMapper(module.to_params())
    before = _panel_metrics(initial, table)
    batches = batches_from_table(table, settings.batch_size)
    history = fit(
        module,
        batches,
        brier_mass_loss,
        settings,
        checkpoint_path=checkpoint_path,
        ema=False,
    )
    fitted = MassMapper(module.to_params())
    after = _panel_metrics(fitted, table)
    return MassCalibrationReport(mapper=fitted, before=before, after=after, history=history)
