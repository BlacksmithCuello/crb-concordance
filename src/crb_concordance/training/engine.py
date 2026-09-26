"""Generic fitting loop for the calibration-stage parameters.

Ref: Sec. 4.2, Algorithm 3 (discovery time updates no parameter, so all fitting is
confined to calibration); Sec. 4.1 (the discount rates are fitted on the
retrospective control panel and never recalibrated during discovery).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from crb_concordance.training.checkpointing import CheckpointMeta, save_checkpoint
from crb_concordance.training.optim import OptimConfig, learning_rate_at, total_steps
from crb_concordance.training.seed import set_seed

LOGGER = logging.getLogger("crb_concordance.training.engine")


class EngineError(ValueError):
    """Raised when the fitting loop is configured inconsistently."""


@dataclass(frozen=True, slots=True)
class StepRecord:
    step: int
    epoch: int
    loss: float
    learning_rate: float
    grad_norm: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
        }


@dataclass(slots=True)
class TrainingHistory:
    """Every step's loss plus the checkpoint written at the end."""

    records: list[StepRecord] = field(default_factory=list)
    checkpoint: CheckpointMeta | None = None

    def losses(self) -> np.ndarray:
        return np.asarray([record.loss for record in self.records], dtype=float)

    @property
    def initial_loss(self) -> float:
        return self.records[0].loss if self.records else float("nan")

    @property
    def final_loss(self) -> float:
        return self.records[-1].loss if self.records else float("nan")

    def decreased(self, *, tolerance: float = 0.0) -> bool:
        return bool(self.losses().size > 1 and self.final_loss < self.initial_loss - tolerance)

    def as_dict(self) -> dict[str, object]:
        return {
            "steps": len(self.records),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "decreased": self.decreased(),
            "records": [record.as_dict() for record in self.records],
            "checkpoint": None if self.checkpoint is None else self.checkpoint.as_dict(),
        }


Batch = tuple[torch.Tensor, ...]


def _parameters(module: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in module.named_parameters()
    }


def _autocast(precision: str) -> Any:
    if precision == "fp16":
        return torch.autocast(device_type="cpu", dtype=torch.float16, enabled=True)
    if precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)
    return torch.autocast(device_type="cpu", enabled=False)


def fit(
    module: torch.nn.Module,
    batches: Sequence[Batch],
    loss_fn: Callable[..., torch.Tensor],
    config: OptimConfig,
    *,
    checkpoint_path: str | Path | None = None,
    ema: bool = True,
) -> TrainingHistory:
    """Train ``module`` on ``batches`` for the configured number of epochs."""

    config.validate()
    if not batches:
        raise EngineError("no batches supplied")
    set_seed(config.seed)
    history = TrainingHistory()
    optimiser = torch.optim.AdamW(
        module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps = total_steps(config, samples=len(batches) * config.effective_batch)
    snapshot: dict[str, torch.Tensor] | None = (
        copy.deepcopy(module.state_dict()) if ema and config.ema_decay > 0.0 else None
    )
    step = 0
    for epoch in range(config.epochs):
        for position, batch in enumerate(batches):
            rate = learning_rate_at(step, steps, config)
            for group in optimiser.param_groups:
                group["lr"] = rate
            with _autocast(config.precision):
                loss = loss_fn(module, *batch)
            if not torch.isfinite(loss):
                raise EngineError(f"non-finite loss at step {step}")
            (loss / config.grad_accum).backward()
            accumulate = (position + 1) % config.grad_accum == 0
            last = position == len(batches) - 1
            grad_norm = 0.0
            if accumulate or last:
                if config.grad_clip_norm > 0.0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(module.parameters(), config.grad_clip_norm)
                    )
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                if snapshot is not None:
                    with torch.no_grad():
                        for name, value in module.state_dict().items():
                            if name in snapshot and value.dtype.is_floating_point:
                                snapshot[name].mul_(config.ema_decay).add_(
                                    value, alpha=1.0 - config.ema_decay
                                )
                            elif name in snapshot:
                                snapshot[name] = value.clone()
            history.records.append(
                StepRecord(
                    step=step,
                    epoch=epoch,
                    loss=float(loss.detach()),
                    learning_rate=rate,
                    grad_norm=grad_norm,
                )
            )
            step += 1
    if snapshot is not None:
        module.load_state_dict(snapshot)
    if checkpoint_path is not None:
        history.checkpoint = save_checkpoint(
            checkpoint_path,
            _parameters(module),
            seed=config.seed,
            step=step,
            config=config.as_dict(),
        )
        LOGGER.info("wrote checkpoint for %d steps", step)
    return history


def evaluate_loss(
    module: torch.nn.Module, batches: Iterable[Batch], loss_fn: Callable[..., torch.Tensor]
) -> float:
    """Mean loss over the supplied batches with gradients disabled."""

    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches:
            total += float(loss_fn(module, *batch))
            count += 1
    if count == 0:
        raise EngineError("no batches to evaluate")
    return total / count
