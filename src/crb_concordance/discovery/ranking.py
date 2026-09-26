"""Ranking of the candidate pool by the pignistic point estimate.

Ref: Sec. 4.1 (the factors of the ranking procedure use r(g) := BetP_g(V));
Sec. 4.2, Algorithm 1 line 10 and Algorithm 3 step 5 (insert g into the pool ranked
by BetP(V) descending, carrying the interval and the trace).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from crb_concordance.belief.mass import BeliefReport


class RankingConfigError(ValueError):
    """Raised when a ranking request is malformed."""


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    rank: int
    report: BeliefReport

    @property
    def candidate(self) -> str:
        return self.report.candidate

    @property
    def pignistic(self) -> float:
        return self.report.pignistic

    @property
    def interval_width(self) -> float:
        return self.report.interval_width

    def as_row(self) -> dict[str, object]:
        row = self.report.as_row()
        row["rank"] = self.rank
        row["flagged_steps"] = sum(1 for _, _, k_mass in self.report.trace if k_mass > 0.0)
        return row


def rank_reports(reports: tuple[BeliefReport, ...]) -> tuple[RankedCandidate, ...]:
    """Descending BetP(V), ties broken by candidate name for reproducibility."""

    ordered = sorted(reports, key=lambda report: (-report.pignistic, report.candidate))
    return tuple(
        RankedCandidate(rank=position, report=report)
        for position, report in enumerate(ordered, start=1)
    )


def rank_lookup(ranked: tuple[RankedCandidate, ...]) -> dict[str, int]:
    return {entry.candidate: entry.rank for entry in ranked}


def concordant(
    ranked: tuple[RankedCandidate, ...], *, width_ceiling: float
) -> tuple[RankedCandidate, ...]:
    """Candidates whose belief interval is narrow enough to call concordant."""

    if width_ceiling < 0.0:
        raise RankingConfigError(f"width ceiling must be non-negative: {width_ceiling!r}")
    return tuple(entry for entry in ranked if entry.interval_width <= width_ceiling)


def discordant(
    ranked: tuple[RankedCandidate, ...], *, width_floor: float
) -> tuple[RankedCandidate, ...]:
    """Candidates whose interval width marks them as cross-modally discordant."""

    if width_floor < 0.0:
        raise RankingConfigError(f"width floor must be non-negative: {width_floor!r}")
    return tuple(entry for entry in ranked if entry.interval_width >= width_floor)


def separability(ranked: tuple[RankedCandidate, ...], *, width_ceiling: float) -> dict[str, float]:
    """How cleanly the interval width separates concordant from discordant profiles."""

    concordant_group = [
        entry.interval_width for entry in concordant(ranked, width_ceiling=width_ceiling)
    ]
    discordant_group = [
        entry.interval_width for entry in discordant(ranked, width_floor=width_ceiling)
    ]
    if not concordant_group or not discordant_group:
        return {
            "auroc": 0.0,
            "concordant": len(concordant_group),
            "discordant": len(discordant_group),
        }
    widths = np.asarray(concordant_group + discordant_group, dtype=float)
    first = len(concordant_group)
    ranks = rankdata(-widths)
    rank_sum = float(np.sum(ranks[:first]))
    auroc = (rank_sum - first * (first + 1) / 2.0) / (first * len(discordant_group))
    return {
        "auroc": auroc,
        "concordant": len(concordant_group),
        "discordant": len(discordant_group),
        "mean_width_concordant": sum(concordant_group) / len(concordant_group),
        "mean_width_discordant": sum(discordant_group) / len(discordant_group),
    }


def top_names(ranked: tuple[RankedCandidate, ...], k: int) -> tuple[str, ...]:
    return tuple(entry.candidate for entry in ranked[:k])


def score_map(ranked: tuple[RankedCandidate, ...]) -> dict[str, float]:
    return {entry.candidate: entry.pignistic for entry in ranked}
