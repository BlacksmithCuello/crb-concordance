"""Shafer discounting of raw modality masses by reliability alpha_k.

Ref: Sec. 4.1, Eq. (2).
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.belief.mass import AgentMass, MassError, MassTriple
from crb_concordance.utils.types import MODALITY_ORDER, Modality


def discount(triple: MassTriple, alpha: float) -> MassTriple:
    """Apply Eq. (2): committed mass is scaled by alpha, the shortfall joins Theta_g."""

    if not 0.0 <= alpha <= 1.0:
        raise MassError(f"discount rate outside [0, 1]: {alpha!r}")
    return MassTriple(
        v=alpha * triple.v,
        n=alpha * triple.n,
        theta=(1.0 - alpha) + alpha * triple.theta,
    )


def undiscount(triple: MassTriple, alpha: float) -> MassTriple:
    """Invert Eq. (2) where the discounted ignorance exceeds the vacuous floor."""

    if not 0.0 < alpha <= 1.0:
        raise MassError(f"discount rate must lie in (0, 1] to invert: {alpha!r}")
    floor = 1.0 - alpha
    if triple.theta + 1e-12 < floor:
        raise MassError("discounted ignorance is below the vacuous floor; not invertible")
    return MassTriple(
        v=triple.v / alpha,
        n=triple.n / alpha,
        theta=(triple.theta - floor) / alpha,
    )


def discount_floor(alpha: float) -> float:
    """The smallest ignorance a modality can carry once its rate is fixed."""

    return 1.0 - alpha


@dataclass(frozen=True, slots=True)
class DiscountRates:
    """The calibrated reliability of every modality, one rate per calibrated fold."""

    rates: dict[Modality, float]

    def __post_init__(self) -> None:
        missing = [k for k in MODALITY_ORDER if k not in self.rates]
        if missing:
            raise MassError(f"missing discount rates for modalities: {[m.value for m in missing]}")
        for modality, rate in self.rates.items():
            if not 0.0 <= rate <= 1.0:
                raise MassError(f"discount rate for {modality.value} outside [0, 1]: {rate!r}")

    def of(self, modality: Modality) -> float:
        return self.rates[modality]

    def as_dict(self) -> dict[str, float]:
        return {modality.value: rate for modality, rate in self.rates.items()}

    def reliability_order(self) -> tuple[Modality, ...]:
        """Descending reliability, the pre-declared order of Algorithm 2."""

        return tuple(
            sorted(MODALITY_ORDER, key=lambda m: (-self.rates[m], MODALITY_ORDER.index(m)))
        )

    def ignorance_floor(self) -> float:
        """Product of the per-modality knowledge gaps, the floor of Proposition 2."""

        product = 1.0
        for modality in MODALITY_ORDER:
            product *= 1.0 - self.rates[modality]
        return product


def discount_all(
    masses: dict[Modality, MassTriple], rates: DiscountRates
) -> dict[Modality, MassTriple]:
    return {modality: discount(triple, rates.of(modality)) for modality, triple in masses.items()}


def order_agent_masses(
    masses: dict[Modality, AgentMass], order: tuple[Modality, ...]
) -> tuple[AgentMass, ...]:
    return tuple(masses[modality] for modality in order)
