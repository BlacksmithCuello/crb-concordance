"""Alternative fusion rules retained as the substitution-tier contrast.

Ref: Sec. 4.5 (the substitution tier replaces the conflict-redistributing rule
with standard Dempster renormalization, naive averaging and Bayesian log-odds).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from crb_concordance.belief.mass import MassTriple
from crb_concordance.utils.numerics import EPS, clip01, logit, sigmoid
from crb_concordance.utils.types import Modality

LOG_ODDS_CLIP = 1e-6


def naive_average(triples: Sequence[MassTriple]) -> MassTriple:
    """Component-wise mean of the modality masses, which stays on the simplex."""

    if not triples:
        raise ValueError("naive averaging needs at least one mass function")
    stacked = np.asarray([triple.as_tuple() for triple in triples], dtype=float)
    mean = stacked.mean(axis=0)
    return MassTriple(v=float(mean[0]), n=float(mean[1]), theta=float(mean[2]))


def naive_sum(triples: Sequence[MassTriple]) -> MassTriple:
    """Sum of the modality masses, renormalised back onto the simplex."""

    if not triples:
        raise ValueError("naive summation needs at least one mass function")
    stacked = np.asarray([triple.as_tuple() for triple in triples], dtype=float)
    total = float(stacked.sum())
    if total <= EPS:
        raise ValueError("naive sum of masses vanishes")
    summed = stacked.sum(axis=0) / total
    return MassTriple(v=float(summed[0]), n=float(summed[1]), theta=float(summed[2]))


def bayesian_log_odds(probabilities: dict[Modality, float], *, prior: float = 0.5) -> float:
    """Log-odds fusion of calibrated per-modality probabilities.

    With four conditionally independent modalities and a common prior this is the
    Bayesian-optimal posterior; the paper keeps it only as a comparator because it
    cannot express ignorance distinctly from negative evidence.
    """

    if not probabilities:
        raise ValueError("log-odds fusion needs at least one modality probability")
    accumulator = logit(prior)
    for probability in probabilities.values():
        accumulator += logit(probability) - logit(prior)
    log_odds = float(np.clip(accumulator, -1.0 / LOG_ODDS_CLIP, 1.0 / LOG_ODDS_CLIP))
    return clip01(float(sigmoid(log_odds)))


def log_odds_triple(probabilities: dict[Modality, float], *, prior: float = 0.5) -> MassTriple:
    """Express the log-odds posterior as a committed triple with zero ignorance."""

    posterior = bayesian_log_odds(probabilities, prior=prior)
    return MassTriple(v=posterior, n=1.0 - posterior, theta=0.0)


def pignistic_probabilities(triples: dict[Modality, MassTriple]) -> dict[Modality, float]:
    return {modality: triple.pignistic() for modality, triple in triples.items()}
