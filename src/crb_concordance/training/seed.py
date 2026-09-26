"""Single seeding utility shared by every entry point.

Ref: Sec. 4.6 (every reported quantity is reproducible from the archived code, and
each resumed run restores the seed recorded in its checkpoint).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


class SeedError(ValueError):
    """Raised when a seed is outside the representable range."""


@dataclass(frozen=True, slots=True)
class SeedState:
    """The seed a run was started with, carried into every checkpoint."""

    seed: int
    deterministic: bool = True

    def as_dict(self) -> dict[str, int | bool]:
        return {"seed": self.seed, "deterministic": self.deterministic}


def set_seed(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed the standard library, NumPy and torch when it is importable."""

    if not 0 <= int(seed) < 2**32:
        raise SeedError(f"seed must lie in [0, 2**32): {seed!r}")
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)
    try:
        import torch
    except ImportError:
        return SeedState(seed=value, deterministic=deterministic)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return SeedState(seed=value, deterministic=deterministic)


def numpy_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def torch_generator(seed: int) -> object | None:
    try:
        import torch
    except ImportError:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
