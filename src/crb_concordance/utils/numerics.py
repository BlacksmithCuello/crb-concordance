"""Deterministic numeric helpers shared by the evidence agents and the statistics layer.

Ref: Sec. 4.6 (determinism and fixed resample counts).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")

EPS = 1e-12


def clip01(value: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(max(value, lo), hi))


def clip01_array(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def logit(probability: float) -> float:
    p = clip01(probability, lo=EPS, hi=1.0 - EPS)
    return float(np.log(p / (1.0 - p)))


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def seed_from_text(text: str, *, modulo: int = 2**31 - 1) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def rng_from(seed: int | str) -> np.random.Generator:
    if isinstance(seed, str):
        return np.random.default_rng(seed_from_text(seed))
    return np.random.default_rng(int(seed))


def rank_average(scores: Sequence[np.ndarray]) -> np.ndarray:
    """Average descending-score ranks across modalities, ties resolved by index order."""

    stacked = np.vstack([np.asarray(s, dtype=float) for s in scores])
    ranks = np.empty_like(stacked)
    for row in range(stacked.shape[0]):
        order = np.argsort(-stacked[row], kind="stable")
        ranks[row, order] = np.arange(stacked.shape[0], dtype=float)
    return ranks.mean(axis=0)


def min_max_scale(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    spread = float(np.max(arr) - np.min(arr))
    if spread <= EPS:
        return np.zeros_like(arr)
    return (arr - float(np.min(arr))) / spread


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    deviation = float(np.std(arr))
    if deviation <= EPS:
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / deviation


def golden_section_minimum(
    objective: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 300,
) -> tuple[float, float]:
    """Derivative-free 1-D minimiser used wherever a scalar rate is fitted."""

    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    f_c, f_d = objective(c), objective(d)
    for _ in range(max_iterations):
        if b - a <= tolerance:
            break
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - inv_phi * (b - a)
            f_c = objective(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + inv_phi * (b - a)
            f_d = objective(d)
    x = 0.5 * (a + b)
    return x, objective(x)


def bootstrap_indices(n: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(resamples, n))


def percentile_interval(samples: np.ndarray, *, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return (
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
    )


def normal_ppf(probability: float) -> float:
    """Acklam's rational approximation, accurate to about 1e-9 over (0, 1)."""

    p = clip01(probability, lo=1e-15, hi=1.0 - 1e-15)
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    lower, upper = 0.02425, 1.0 - 0.02425
    if p < lower:
        q = np.sqrt(-2.0 * np.log(p))
        return float(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > upper:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return float(
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return float(
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def stable_softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / max(temperature, EPS)
    shifted = scaled - float(np.max(scaled))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))
