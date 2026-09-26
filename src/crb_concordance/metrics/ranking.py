"""Retrieval metrics of the temporal-holdout rediscovery benchmark.

Ref: Sec. 4.4 (Recall@k, Precision@k, mean reciprocal rank and the fold-level hit
rate reported individually so a control expected to rank lower is not averaged away).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RankingError(ValueError):
    """Raised when a ranking cannot be scored."""


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    average_precision: float
    cutoff: int
    positives: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "average_precision": self.average_precision,
            "cutoff": self.cutoff,
            "positives": self.positives,
        }


def rank_order(scores: dict[str, float]) -> tuple[str, ...]:
    """Descending score order with ties broken by name for determinism."""

    return tuple(sorted(scores, key=lambda name: (-scores[name], name)))


def recall_at_k(ranked: tuple[str, ...], positives: set[str], k: int) -> float:
    if not positives:
        raise RankingError("recall is undefined without a positive set")
    if k < 1:
        raise RankingError(f"cutoff must be positive: {k!r}")
    recovered = sum(1 for name in ranked[:k] if name in positives)
    return recovered / len(positives)


def precision_at_k(ranked: tuple[str, ...], positives: set[str], k: int) -> float:
    if k < 1:
        raise RankingError(f"cutoff must be positive: {k!r}")
    window = ranked[:k]
    if not window:
        return 0.0
    return sum(1 for name in window if name in positives) / len(window)


def mean_reciprocal_rank(ranked: tuple[str, ...], positives: set[str]) -> float:
    for position, name in enumerate(ranked, start=1):
        if name in positives:
            return 1.0 / position
    return 0.0


def average_precision(ranked: tuple[str, ...], positives: set[str]) -> float:
    if not positives:
        raise RankingError("average precision is undefined without a positive set")
    hits = 0
    total = 0.0
    for position, name in enumerate(ranked, start=1):
        if name in positives:
            hits += 1
            total += hits / position
    return total / len(positives)


def hit_rate(ranked: tuple[str, ...], target: str, k: int) -> float:
    """Fold-level indicator for one named control, reported on its own."""

    return 1.0 if target in ranked[:k] else 0.0


def evaluate_ranking(
    scores: dict[str, float], positives: set[str], *, k: int = 10
) -> RankingMetrics:
    ranked = rank_order(scores)
    return RankingMetrics(
        recall_at_k=recall_at_k(ranked, positives, k),
        precision_at_k=precision_at_k(ranked, positives, k),
        mean_reciprocal_rank=mean_reciprocal_rank(ranked, positives),
        average_precision=average_precision(ranked, positives),
        cutoff=k,
        positives=len(positives),
    )


def recall_curve(
    scores: dict[str, float], positives: set[str], cutoffs: tuple[int, ...]
) -> dict[int, float]:
    ranked = rank_order(scores)
    return {k: recall_at_k(ranked, positives, k) for k in cutoffs}


def fold_averaged_recall(
    per_fold_scores: tuple[dict[str, float], ...], positives: set[str], *, k: int = 10
) -> float:
    if not per_fold_scores:
        raise RankingError("no folds supplied")
    values = [recall_at_k(rank_order(fold), positives, k) for fold in per_fold_scores]
    return float(np.mean(values))


def top_k(scores: dict[str, float], k: int) -> tuple[str, ...]:
    return rank_order(scores)[:k]


def rank_of(scores: dict[str, float], name: str) -> int | None:
    ranked = rank_order(scores)
    for position, candidate in enumerate(ranked, start=1):
        if candidate == name:
            return position
    return None
