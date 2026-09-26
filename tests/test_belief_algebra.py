"""Belief-function algebra: every expectation is recomputed by an independent route."""

from __future__ import annotations

import numpy as np
import pytest

from crb_concordance.belief.combine import (
    combine_pair,
    combine_pair_by_enumeration,
    combine_sequential,
    dempster_pair,
)
from crb_concordance.belief.discount import discount, undiscount
from crb_concordance.belief.fusion import bayesian_log_odds, naive_average, naive_sum
from crb_concordance.belief.guarantees import (
    conflict_monotonicity_scan,
    pignistic_bias_floor,
    pignistic_convergence_scan,
    proposition_summary,
    zero_conflict_limit,
)
from crb_concordance.belief.mass import MassError, MassTriple, from_posterior, vacuous
from crb_concordance.belief.trace import PLANNING_FLAG_THRESHOLD, build_trace

GENERATOR = np.random.default_rng(4)


def _random_triple() -> MassTriple:
    values = GENERATOR.dirichlet([1.0, 1.0, 1.0])
    return MassTriple(v=float(values[0]), n=float(values[1]), theta=float(values[2]))


def test_simplex_validation_rejects_off_simplex_triples() -> None:
    with pytest.raises(MassError):
        MassTriple(v=0.5, n=0.5, theta=0.5)
    with pytest.raises(MassError):
        MassTriple(v=-0.1, n=0.6, theta=0.5)


def test_discount_matches_the_hand_computed_terms() -> None:
    triple = MassTriple(v=0.6, n=0.3, theta=0.1)
    discounted = discount(triple, 0.5)
    assert discounted.v == pytest.approx(0.3)
    assert discounted.n == pytest.approx(0.15)
    assert discounted.theta == pytest.approx(0.5 + 0.5 * 0.1)


def test_discount_round_trip() -> None:
    for _ in range(20):
        triple = _random_triple()
        recovered = undiscount(discount(triple, 0.42), 0.42)
        assert recovered.as_tuple() == pytest.approx(triple.as_tuple(), abs=1e-12)


def test_unit_rate_discount_is_the_identity() -> None:
    triple = _random_triple()
    assert discount(triple, 1.0).as_tuple() == pytest.approx(triple.as_tuple())
    assert discount(triple, 0.0) == vacuous()


def test_combination_agrees_with_explicit_focal_products() -> None:
    for _ in range(50):
        left, right = _random_triple(), _random_triple()
        shortcut, k_shortcut = combine_pair(left, right)
        enumerated, k_enumerated = combine_pair_by_enumeration(left, right)
        assert shortcut.as_tuple() == pytest.approx(enumerated.as_tuple(), abs=1e-14)
        assert k_shortcut == pytest.approx(k_enumerated, abs=1e-14)


def test_combination_step_matches_the_closed_form_terms() -> None:
    left = MassTriple(v=0.5, n=0.2, theta=0.3)
    right = MassTriple(v=0.1, n=0.6, theta=0.3)
    combined, k_step = combine_pair(left, right)
    assert k_step == pytest.approx(0.5 * 0.6 + 0.2 * 0.1)
    assert combined.v == pytest.approx(0.5 * 0.1 + 0.5 * 0.3 + 0.3 * 0.1)
    assert combined.n == pytest.approx(0.2 * 0.6 + 0.2 * 0.3 + 0.3 * 0.6)
    assert combined.theta == pytest.approx(0.3 * 0.3 + k_step)


def test_combination_conserves_mass() -> None:
    for _ in range(50):
        triples = [_random_triple() for _ in range(4)]
        result = combine_sequential(triples)
        assert sum(result.triple.as_tuple()) == pytest.approx(1.0, abs=1e-12)
        assert result.total_conflict == pytest.approx(sum(step.k_mass for step in result.steps))


def test_combination_is_order_sensitive() -> None:
    triples = [
        MassTriple(v=0.6, n=0.3, theta=0.1),
        MassTriple(v=0.2, n=0.7, theta=0.1),
        MassTriple(v=0.5, n=0.4, theta=0.1),
    ]
    forward = combine_sequential(triples).pignistic
    reversed_order = combine_sequential(list(reversed(triples))).pignistic
    assert forward != pytest.approx(reversed_order)


def test_interval_quantities_are_the_closed_forms() -> None:
    triple = MassTriple(v=0.42, n=0.31, theta=0.27)
    assert triple.belief() == pytest.approx(0.42)
    assert triple.plausibility() == pytest.approx(0.69)
    assert triple.pignistic() == pytest.approx(0.555)
    assert triple.interval_width() == pytest.approx(0.27)
    assert triple.plausibility() - triple.belief() == pytest.approx(triple.theta)


def test_dempster_renormalisation_shrinks_the_interval() -> None:
    left = MassTriple(v=0.5, n=0.4, theta=0.1)
    right = MassTriple(v=0.4, n=0.5, theta=0.1)
    ours, _ = combine_pair(left, right)
    dempster, k_step = dempster_pair(left, right)
    assert dempster.interval_width() < ours.interval_width()
    assert dempster.pignistic() >= ours.pignistic()
    assert k_step > 0.0


def test_trace_flags_steps_above_the_threshold() -> None:
    triples = [MassTriple(v=0.9, n=0.05, theta=0.05), MassTriple(v=0.05, n=0.9, theta=0.05)]
    result = combine_sequential(triples, ("a", "b"))
    trace = build_trace(result, PLANNING_FLAG_THRESHOLD)
    assert trace.flagged()
    assert trace.total == pytest.approx(result.total_conflict)
    assert trace.dominant() is not None


def test_naive_fusion_hand_computed() -> None:
    triples = [MassTriple(0.6, 0.2, 0.2), MassTriple(0.2, 0.6, 0.2)]
    averaged = naive_average(triples)
    assert averaged.as_tuple() == pytest.approx((0.4, 0.4, 0.2))
    summed = naive_sum(triples)
    assert summed.as_tuple() == pytest.approx((0.4, 0.4, 0.2))


def test_log_odds_fusion_hand_computed() -> None:
    from crb_concordance.utils.types import Modality

    posterior = bayesian_log_odds({Modality.KG: 0.75, Modality.DEP: 0.75}, prior=0.5)
    expected = 1.0 / (1.0 + np.exp(-2.0 * np.log(0.75 / 0.25)))
    assert posterior == pytest.approx(expected)


def test_from_posterior_splits_the_committed_mass() -> None:
    triple = from_posterior(0.25, ignorance=0.2)
    assert triple.v == pytest.approx(0.2)
    assert triple.n == pytest.approx(0.6)
    assert triple.theta == pytest.approx(0.2)


def test_proposition_one_width_is_monotone_in_conflict(learned_rates) -> None:
    scan = conflict_monotonicity_scan(learned_rates)
    assert scan.monotone_non_decreasing()
    assert scan.width[-1] > scan.width[0]
    assert scan.total_conflict[-1] > scan.total_conflict[0]


def test_proposition_two_bias_equals_half_the_knowledge_gap_product(learned_rates) -> None:
    floor = pignistic_bias_floor(learned_rates)
    limit, bias = zero_conflict_limit(learned_rates, target=True)
    assert floor == pytest.approx(0.5 * learned_rates.ignorance_floor())
    assert limit == pytest.approx(1.0 - floor)
    assert bias == pytest.approx(floor)
    conjugate_limit, conjugate_bias = zero_conflict_limit(learned_rates, target=False)
    assert conjugate_limit == pytest.approx(floor)
    assert conjugate_bias == pytest.approx(floor)


def test_proposition_two_convergence_reaches_the_floor(learned_rates) -> None:
    scan = pignistic_convergence_scan(learned_rates)
    assert scan.approaches_limit(1e-5)
    assert scan.converges_to_floor(1e-5)
    assert scan.pignistic[-1] == pytest.approx(scan.limit, abs=1e-5)


def test_proposition_summary_reports_both_principles(learned_rates) -> None:
    summary = proposition_summary(learned_rates)
    assert set(summary) == {"proposition_1", "proposition_2", "floor_vanishes"}
    assert all(check.holds for check in summary.values())


def test_floor_vanishes_when_a_rate_saturates(learned_rates) -> None:
    from crb_concordance.belief.discount import DiscountRates
    from crb_concordance.utils.types import MODALITY_ORDER

    saturated = DiscountRates({modality: 1.0 for modality in MODALITY_ORDER})
    assert pignistic_bias_floor(saturated) == 0.0
    assert zero_conflict_limit(saturated)[0] == pytest.approx(1.0)
    assert proposition_summary(saturated)["floor_vanishes"].holds
