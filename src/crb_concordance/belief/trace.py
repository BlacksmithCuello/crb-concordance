"""Auditable conflict trace over the sequential combination.

Ref: Sec. 4.1 (conflict-mass trace and the flagging threshold of
Supplementary Table S1); Sec. 4.2, Algorithm 2 lines 8-10.
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.belief.combine import CombinationResult

PLANNING_FLAG_THRESHOLD = 0.05
MIN_PAIR_STEPS = 2


@dataclass(frozen=True, slots=True)
class TraceEntry:
    """One recorded pairwise conflict term and whether it crossed the threshold."""

    step_index: int
    pair: str
    k_mass: float
    flagged: bool

    def as_row(self) -> dict[str, float | int | str | bool]:
        return {
            "step": self.step_index,
            "pair": self.pair,
            "k_mass": self.k_mass,
            "flagged": self.flagged,
        }


@dataclass(frozen=True, slots=True)
class ConflictTrace:
    """The set of pairwise conflicts that drove the final interval width."""

    threshold: float
    entries: tuple[TraceEntry, ...]

    @property
    def total(self) -> float:
        return float(sum(entry.k_mass for entry in self.entries))

    def flagged(self) -> tuple[TraceEntry, ...]:
        return tuple(entry for entry in self.entries if entry.flagged)

    def dominant(self) -> TraceEntry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda entry: entry.k_mass)

    def as_rows(self) -> list[dict[str, float | int | str | bool]]:
        return [entry.as_row() for entry in self.entries]


def build_trace(
    result: CombinationResult, threshold: float = PLANNING_FLAG_THRESHOLD
) -> ConflictTrace:
    if threshold < 0.0:
        raise ValueError(f"conflict-trace threshold must be non-negative: {threshold!r}")
    entries = tuple(
        TraceEntry(
            step_index=step.step_index,
            pair=f"{step.left} x {step.right}",
            k_mass=step.k_mass,
            flagged=step.k_mass >= threshold,
        )
        for step in result.steps
    )
    return ConflictTrace(threshold=threshold, entries=entries)


def trace_summary(trace: ConflictTrace) -> dict[str, float | int | str | None]:
    dominant = trace.dominant()
    return {
        "threshold": trace.threshold,
        "steps": len(trace.entries),
        "flagged_steps": len(trace.flagged()),
        "total_conflict": trace.total,
        "dominant_pair": None if dominant is None else dominant.pair,
        "dominant_mass": 0.0 if dominant is None else dominant.k_mass,
    }
