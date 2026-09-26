"""Belief-mass triples over the two-element frame Theta_g = {V, not V}.

Ref: Sec. 4.1, Eq. (1) and Eq. (4); Sec. 4.2, Algorithm 2 line 12.
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.utils.types import Modality

MASS_TOLERANCE = 1e-9


class MassError(ValueError):
    """Raised when a mass triple violates the simplex constraint of Eq. (1)."""


@dataclass(frozen=True, slots=True)
class MassTriple:
    """A mass function on Theta_g restricted to its three focal elements.

    ``v`` is m(V), ``n`` is m(not V) and ``theta`` is m(Theta_g); Eq. (1) requires
    v + n + theta = 1, so theta is the genuinely uncommitted mass.
    """

    v: float
    n: float
    theta: float

    def __post_init__(self) -> None:
        for label, value in (("v", self.v), ("n", self.n), ("theta", self.theta)):
            if value < -MASS_TOLERANCE:
                raise MassError(f"negative mass on {label}: {value!r}")
        total = self.v + self.n + self.theta
        if abs(total - 1.0) > 1e-7:
            raise MassError(f"mass triple does not sum to one: {total!r}")

    @property
    def committed(self) -> float:
        return self.v + self.n

    @property
    def ignorance(self) -> float:
        return self.theta

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.v, self.n, self.theta)

    def is_vacuous(self, tolerance: float = MASS_TOLERANCE) -> bool:
        return abs(self.theta - 1.0) <= tolerance

    def belief(self) -> float:
        return self.v

    def plausibility(self) -> float:
        return self.v + self.theta

    def pignistic(self) -> float:
        """BetP_g(V) of Eq. (4)."""

        return self.v + 0.5 * self.theta

    def interval_width(self) -> float:
        """Pl_g(V) - Bel_g(V), equal to the uncommitted mass m(Theta_g)."""

        return self.theta


def vacuous() -> MassTriple:
    """Total ignorance: m(Theta_g) = 1, the mass of an absent modality."""

    return MassTriple(v=0.0, n=0.0, theta=1.0)


def certain_v() -> MassTriple:
    return MassTriple(v=1.0, n=0.0, theta=0.0)


def certain_not_v() -> MassTriple:
    return MassTriple(v=0.0, n=1.0, theta=0.0)


def from_posterior(probability: float, *, ignorance: float = 0.0) -> MassTriple:
    """Build a triple whose committed mass is split by a calibrated probability."""

    if not 0.0 <= ignorance <= 1.0:
        raise MassError(f"ignorance outside [0, 1]: {ignorance!r}")
    p = min(max(float(probability), 0.0), 1.0)
    committed = 1.0 - ignorance
    return MassTriple(v=committed * p, n=committed * (1.0 - p), theta=ignorance)


@dataclass(frozen=True, slots=True)
class AgentMass:
    """One modality's mass for one candidate, with its calibration provenance."""

    modality: Modality
    triple: MassTriple
    discount_rate: float
    calibration_fold: int
    evidence_available: bool = True


@dataclass(frozen=True, slots=True)
class BeliefReport:
    """Bel(V), Pl(V), BetP(V) and the interval width for one candidate."""

    candidate: str
    belief: float
    plausibility: float
    pignistic: float
    total_conflict: float
    trace: tuple[tuple[int, str, float], ...]

    @property
    def interval_width(self) -> float:
        return self.plausibility - self.belief

    def as_row(self) -> dict[str, float | str]:
        return {
            "candidate": self.candidate,
            "belief": self.belief,
            "plausibility": self.plausibility,
            "pignistic": self.pignistic,
            "interval_width": self.interval_width,
            "total_conflict": self.total_conflict,
        }
