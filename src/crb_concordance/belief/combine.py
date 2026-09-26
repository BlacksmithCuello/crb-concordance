"""Conflict-redistributing sequential combination.

Ref: Sec. 4.1, Eq. (3); Sec. 4.2, Algorithm 2; the standard Dempster
renormalization retained only as the substitution-tier contrast.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from crb_concordance.belief.mass import MassTriple
from crb_concordance.utils.types import MODALITY_ORDER, Modality

FOCAL_LABELS: tuple[str, ...] = ("V", "NOT_V", "THETA")


class CombinationError(ValueError):
    """Raised when a combination is undefined, as in total Dempster conflict."""


@dataclass(frozen=True, slots=True)
class ConflictStep:
    """One pairwise conflict term K_step of Eq. (3)."""

    step_index: int
    left: str
    right: str
    k_mass: float


@dataclass(frozen=True, slots=True)
class CombinationResult:
    """The combined triple plus the auditable conflict trace."""

    triple: MassTriple
    total_conflict: float
    steps: tuple[ConflictStep, ...]

    @property
    def belief(self) -> float:
        return self.triple.belief()

    @property
    def plausibility(self) -> float:
        return self.triple.plausibility()

    @property
    def pignistic(self) -> float:
        return self.triple.pignistic()

    @property
    def interval_width(self) -> float:
        return self.triple.interval_width()

    def trace_rows(self) -> list[tuple[int, str, float]]:
        return [(step.step_index, f"{step.left}+{step.right}", step.k_mass) for step in self.steps]


def combine_pair(left: MassTriple, right: MassTriple) -> tuple[MassTriple, float]:
    """One sequential step of Algorithm 2 lines 3-6.

    Returns the combined triple and K_step. The conflict term is routed into
    m(Theta_g) and is never divided out, which is what keeps the Zadeh paradox
    out of the interval.
    """

    v = left.v * right.v + left.v * right.theta + left.theta * right.v
    n = left.n * right.n + left.n * right.theta + left.theta * right.n
    k_step = left.v * right.n + left.n * right.v
    return MassTriple(v=v, n=n, theta=left.theta * right.theta + k_step), k_step


def combine_pair_by_enumeration(left: MassTriple, right: MassTriple) -> tuple[MassTriple, float]:
    """Independent restatement of ``combine_pair`` over the explicit focal products."""

    masses_left = dict(zip(FOCAL_LABELS, left.as_tuple(), strict=True))
    masses_right = dict(zip(FOCAL_LABELS, right.as_tuple(), strict=True))
    accumulated = dict.fromkeys(FOCAL_LABELS, 0.0)
    conflict = 0.0
    for (label_l, mass_l), (label_r, mass_r) in product(masses_left.items(), masses_right.items()):
        if label_l == "THETA":
            target = label_r
        elif label_r == "THETA" or label_l == label_r:
            target = label_l
        else:
            target = "THETA"
            conflict += mass_l * mass_r
        accumulated[target] += mass_l * mass_r
    return (
        MassTriple(v=accumulated["V"], n=accumulated["NOT_V"], theta=accumulated["THETA"]),
        conflict,
    )


def combine_sequential(
    triples: Sequence[MassTriple],
    labels: Sequence[str] | None = None,
) -> CombinationResult:
    """Fold the whole reliability-ordered sequence, accumulating the conflict trace."""

    if len(triples) < 2:
        raise CombinationError("sequential combination needs at least two mass functions")
    names = list(labels) if labels is not None else [f"m{i + 1}" for i in range(len(triples))]
    current = triples[0]
    carried = names[0]
    steps: list[ConflictStep] = []
    total = 0.0
    for offset, triple in enumerate(triples[1:], start=1):
        current, k_step = combine_pair(current, triple)
        total += k_step
        steps.append(
            ConflictStep(
                step_index=offset + 1,
                left=carried,
                right=names[offset],
                k_mass=k_step,
            )
        )
        carried = f"{carried}|{names[offset]}"
    return CombinationResult(triple=current, total_conflict=total, steps=tuple(steps))


def combine_modality_masses(
    masses_by_modality: dict[Modality, MassTriple],
    order: Sequence[Modality] | None = None,
) -> CombinationResult:
    sequence_order = list(order) if order is not None else list(MODALITY_ORDER)
    missing = [modality for modality in sequence_order if modality not in masses_by_modality]
    if missing:
        raise CombinationError(f"no mass supplied for modalities: {[m.value for m in missing]}")
    triples = [masses_by_modality[modality] for modality in sequence_order]
    labels = [modality.value for modality in sequence_order]
    return combine_sequential(triples, labels)


def dempster_pair(left: MassTriple, right: MassTriple) -> tuple[MassTriple, float]:
    """Classical Dempster combination with conflict renormalized away."""

    _, k_step = combine_pair_by_enumeration(left, right)
    if k_step >= 1.0 - 1e-15:
        raise CombinationError("total conflict: Dempster renormalization is undefined")
    raw_v = left.v * right.v + left.v * right.theta + left.theta * right.v
    raw_n = left.n * right.n + left.n * right.theta + left.theta * right.n
    raw_theta = left.theta * right.theta
    scale = 1.0 / (1.0 - k_step)
    return (
        MassTriple(v=raw_v * scale, n=raw_n * scale, theta=raw_theta * scale),
        k_step,
    )


def dempster_sequential(
    triples: Sequence[MassTriple], labels: Sequence[str] | None = None
) -> CombinationResult:
    names = list(labels) if labels is not None else [f"m{i + 1}" for i in range(len(triples))]
    current = triples[0]
    carried = names[0]
    steps: list[ConflictStep] = []
    total = 0.0
    for offset, triple in enumerate(triples[1:], start=1):
        current, k_step = dempster_pair(current, triple)
        total += k_step
        steps.append(
            ConflictStep(
                step_index=offset + 1,
                left=carried,
                right=names[offset],
                k_mass=k_step,
            )
        )
        carried = f"{carried}|{names[offset]}"
    return CombinationResult(triple=current, total_conflict=total, steps=tuple(steps))
