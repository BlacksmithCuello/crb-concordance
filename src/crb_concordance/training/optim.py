"""Optimiser, learning-rate schedule and gradient-clipping configuration.

Ref: Sec. 4.1 and Eq. (2) (the discount rates are fitted once per calibration fold and
never updated during discovery, so any optimisation is confined to calibration time);
Sec. 4.6 (the pre-specified analysis is what the released defaults must reproduce).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ScheduleError(ValueError):
    """Raised when a schedule is configured inconsistently."""


@dataclass(frozen=True, slots=True)
class OptimConfig:
    """Everything the fitting loop needs about optimisation."""

    learning_rate: float = 0.05
    weight_decay: float = 0.0
    epochs: int = 200
    batch_size: int = 32
    grad_accum: int = 1
    warmup_fraction: float = 0.1
    scheduler: str = "cosine"
    min_learning_rate: float = 1e-5
    grad_clip_norm: float = 1.0
    precision: str = "fp32"
    ema_decay: float = 0.0
    seed: int = 20260101

    def validate(self) -> None:
        if self.learning_rate <= 0.0:
            raise ScheduleError(f"learning rate must be positive: {self.learning_rate!r}")
        if self.epochs < 1:
            raise ScheduleError(f"epochs must be positive: {self.epochs!r}")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ScheduleError("batch size and gradient accumulation must be positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ScheduleError(f"warmup fraction must lie in [0, 1): {self.warmup_fraction!r}")
        if self.scheduler not in ("constant", "cosine", "linear"):
            raise ScheduleError(f"unsupported scheduler: {self.scheduler!r}")
        if self.precision not in ("fp32", "fp16", "bf16", "tf32"):
            raise ScheduleError(f"unsupported precision: {self.precision!r}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ScheduleError(f"EMA decay must lie in [0, 1): {self.ema_decay!r}")

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def as_dict(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "effective_batch": self.effective_batch,
            "warmup_fraction": self.warmup_fraction,
            "scheduler": self.scheduler,
            "min_learning_rate": self.min_learning_rate,
            "grad_clip_norm": self.grad_clip_norm,
            "precision": self.precision,
            "ema_decay": self.ema_decay,
            "seed": self.seed,
        }


def learning_rate_at(step: int, total_steps: int, config: OptimConfig) -> float:
    """Warmup followed by the declared decay of the configured schedule."""

    config.validate()
    if total_steps < 1:
        raise ScheduleError("total steps must be positive")
    if step < 0:
        raise ScheduleError(f"step must be non-negative: {step!r}")
    warmup_steps = int(config.warmup_fraction * total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return float(config.learning_rate * (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = float(np.clip(progress, 0.0, 1.0))
    if config.scheduler == "constant":
        return float(config.learning_rate)
    if config.scheduler == "linear":
        return float(
            config.learning_rate + progress * (config.min_learning_rate - config.learning_rate)
        )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine
    )


def schedule_curve(config: OptimConfig, *, steps_per_epoch: int) -> np.ndarray:
    total = config.epochs * steps_per_epoch
    return np.asarray([learning_rate_at(step, total, config) for step in range(total)], dtype=float)


def total_steps(config: OptimConfig, *, samples: int) -> int:
    config.validate()
    if samples < 1:
        raise ScheduleError("at least one sample is required")
    batches = math.ceil(samples / config.effective_batch)
    return max(1, batches * config.epochs)
