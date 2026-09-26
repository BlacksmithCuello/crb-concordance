"""Benchmark table, scoring operators, metric primitives and the statistics plan."""

from __future__ import annotations

import numpy as np
import pytest

from crb_concordance.benchmark.baselines.operators import (
    OPERATORS,
    Aggregation,
    aggregate,
    operator_for,
    perturbed,
    score_pool,
    unconstrained_ensemble_operator,
    validate_bindings,
)
from crb_concordance.benchmark.pool import (
    SplitError,
    assemble_pool,
    build_folds,
    cutoff_control_report,
    pool_size_summary,
)
from crb_concordance.benchmark.registry import (
    DECLARED_BASELINES,
    DECLARED_FUSION_CONTROLS,
    Family,
    all_row_names,
    family_counts,
    family_rows,
    original_benchmark_rows,
    research_rows,
    validate_registry,
)
from crb_concordance.evaluation.reporting import (
    TABLE_TWO_METHODS,
    pending_row,
    summarise_seed_runs,
    table_one_snapshot,
    table_two_snapshot,
    verdict_counts,
)
from crb_concordance.graph.paths import PathFinder
from crb_concordance.metrics.calibration import (
    CalibrationMetricError,
    brier_score,
    calibration_summary,
    expected_calibration_error,
    overconfidence,
    reliability_curve,
)
from crb_concordance.metrics.comparison import (
    bootstrap_mean_interval,
    cluster_bootstrap_difference,
    cochran_q,
    cohens_kappa,
    paired_bootstrap_difference,
    paired_gap_summary,
    seed_level_summary,
)
from crb_concordance.metrics.provenance import (
    audit_citations,
    citation_key,
    hallucinated_edge_rate,
    ledger_index,
    ledger_key,
    provenance_grounding_rate,
    unsupported_ledger_fields,
)
from crb_concordance.metrics.ranking import (
    average_precision,
    evaluate_ranking,
    fold_averaged_recall,
    hit_rate,
    mean_reciprocal_rank,
    precision_at_k,
    rank_order,
    recall_at_k,
    recall_curve,
)
from crb_concordance.simulation.design import SimulationCell, SimulationDesign
from crb_concordance.simulation.evidence_model import summarise_pool
from crb_concordance.stats.delong import midrank, paired_comparison, roc_estimate
from crb_concordance.stats.gee import (
    GeeError,
    arm_gain_report,
    fit_gee_logistic,
    reader_study_design,
)
from crb_concordance.stats.multiplicity import (
    adjusted_never_below_raw,
    benjamini_hochberg,
    family_plan,
    holm_bonferroni,
    holm_rejects_only_uncorrected_rejections,
)
from crb_concordance.stats.power import (
    PowerError,
    SizingInputs,
    accrual_range_covers,
    analytic_minimum_records,
    hanley_mcneil_variance,
    normal_cdf,
    paired_variance,
    power_at_size,
    prospective_sizing,
)
from crb_concordance.stats.survival import (
    SurvivalError,
    concordance_index,
    cumulative_incidence,
    fit_competing_risks,
    fit_cox,
    site_gap_summary,
)
from crb_concordance.utils.types import MODALITY_ORDER


def test_registry_counts_match_the_caption() -> None:
    payload = validate_registry()
    assert payload["baselines"] == DECLARED_BASELINES == 37
    assert payload["fusion_controls"] == DECLARED_FUSION_CONTROLS == 2
    assert payload["table_rows_including_proposed"] == 40
    assert len(all_row_names()) == 39
    assert sum(family_counts().values()) == 39


def test_every_family_is_present_with_its_declared_size() -> None:
    counts = family_counts()
    assert counts[Family.MULTI_AGENT.value] == 10
    assert counts[Family.GRAPH_EMBEDDING.value] == 4
    assert counts[Family.RETRIEVAL.value] == 8
    assert counts[Family.MECHANISTIC.value] == 7
    assert counts[Family.SINGLE_LLM.value] == 4
    assert counts[Family.TRANSCRIPTOMIC.value] == 4
    assert counts[Family.FUSION_CONTROL.value] == 2


def test_reference_numbers_are_transcribed() -> None:
    by_name = {row.name: row for row in family_rows(Family.MULTI_AGENT)}
    assert by_name["AI co-scientist (Gemini-family multi-agent tournament)"].reference == 7
    assert by_name["Virtual Lab (LLM-PI plus specialist-agent team)"].reference == 9
    flux_rows = {row.name: row for row in family_rows(Family.MECHANISTIC)}
    assert flux_rows["Classical flux-balance analysis, unaugmented"].reference == 63
    assert flux_rows["FluxGAT-style flux-sampling-plus-GNN hybrid, gene level"].reference == 37


def test_rows_without_a_reference_are_marked_as_such() -> None:
    unreferenced = [row for row in research_rows() if row.reference is None]
    assert unreferenced
    assert all(not row.reimplemented for row in original_benchmark_rows())
    assert "flux agent in isolation" in " ".join(row.name for row in original_benchmark_rows())


def test_operators_are_bound_to_every_row() -> None:
    payload = validate_bindings()
    assert payload["bound_rows"] == len(OPERATORS) == 39
    assert payload["distinct_aggregations"] >= 6
    assert payload["distinct_modality_sets"] >= 5
    assert (
        operator_for("HAN-style heterogeneous attention network").family is Family.GRAPH_EMBEDDING
    )


def test_unconstrained_ensemble_operator_is_declared() -> None:
    operator = unconstrained_ensemble_operator()
    assert operator.aggregation is Aggregation.RANK_MEAN
    assert not operator.uses_calibration
    assert operator.modalities == MODALITY_ORDER
    assert operator.perturbation_scale == 0.0


def test_aggregations_are_hand_computable() -> None:
    values = np.asarray([0.2, 0.6, 0.8, 1.0])
    assert aggregate(values, Aggregation.MEAN) == pytest.approx(0.65)
    assert aggregate(values, Aggregation.MAX) == pytest.approx(1.0)
    assert aggregate(values, Aggregation.MIN) == pytest.approx(0.2)
    assert aggregate(values, Aggregation.MEDIAN) == pytest.approx(0.7)
    expected_tournament = 0.6 * 1.0 + 0.4 * float(np.mean([0.8, 0.6, 0.2]))
    assert aggregate(values, Aggregation.TOURNAMENT) == pytest.approx(expected_tournament)
    expected_reviewer = 0.6 * 0.2 + 0.4 * float(np.mean([0.6, 0.8, 1.0]))
    assert aggregate(values, Aggregation.REVIEWER) == pytest.approx(expected_reviewer)
    assert aggregate(values, Aggregation.PRODUCT) == pytest.approx(float(np.prod(values) ** 0.25))


def test_perturbation_is_deterministic_and_bounded() -> None:
    first = perturbed(0.5, scale=0.2, seed=3, salt="row")
    second = perturbed(0.5, scale=0.2, seed=3, salt="row")
    other = perturbed(0.5, scale=0.2, seed=4, salt="row")
    assert first == second
    assert first != other
    assert 0.0 <= first <= 1.0
    assert perturbed(0.5, scale=0.0, seed=1, salt="row") == 0.5


def test_operator_scores_cover_the_pool(learned_rates, small_pool) -> None:
    summaries = summarise_pool(small_pool, learned_rates)
    operator = OPERATORS[0]
    scores = score_pool(summaries, operator, seed=1)
    assert set(scores) == {summary.gene for summary in summaries}
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_pool_folds_and_leakage_are_reported(substrate, control_panel) -> None:
    folds = build_folds(substrate)
    assert len(folds) == 5
    assert all(fold.size == len(substrate.candidate_pool) for fold in folds)
    assert all(fold.held_out for fold in folds)

    finder = PathFinder(substrate.graph, max_hops=2)
    pool = assemble_pool(substrate, control_panel, finder)
    assert len(pool.candidates) == len(substrate.candidate_pool)
    assert set(pool.positives) | set(pool.negatives) <= set(pool.candidates)
    assert len(pool.folds) == 5
    assert set(pool.leakage) == {"cutoff", "circularity", "held_out_direct_edges"}
    assert pool.positive_set() == {"SLC2A1", "SLC16A1"}

    payload = cutoff_control_report(substrate)
    assert payload["removed_edges"] > 0
    summary = pool_size_summary(substrate)
    assert summary["metabolic_edges"] > 0
    assert summary["candidates"] == len(substrate.candidate_pool)


def test_fold_request_of_one_is_rejected(substrate) -> None:
    with pytest.raises(SplitError):
        build_folds(substrate, folds=1)


def test_table_one_snapshot_keeps_cells_pending() -> None:
    snapshot = table_one_snapshot()
    assert snapshot["registry"]["baselines"] == 37
    assert len(snapshot["rows"]) == 40
    assert all(not row["executed"] for row in snapshot["rows"])
    assert all(
        value == "[pending]" for row in snapshot["rows"] for value in row["primary"].values()
    )


def test_table_one_snapshot_accepts_executed_rows() -> None:
    snapshot = table_one_snapshot({"BM25 sparse lexical retrieval": {"Recall@10": 0.1}})
    executed = [row for row in snapshot["rows"] if row["executed"]]
    assert len(executed) == 1
    assert executed[0]["primary"]["Recall@10"] == 0.1
    assert executed[0]["primary"]["MRR"] == "[pending]"


def test_table_two_snapshot_is_pending() -> None:
    snapshot = table_two_snapshot()
    assert snapshot["executed"] is False
    assert len(snapshot["rows"]) == len(TABLE_TWO_METHODS)
    assert all(row["value"] == "[pending]" for row in snapshot["rows"])


def test_pending_row_shape() -> None:
    row = pending_row("x", family="f", reference=1)
    assert row["method"] == "x"
    assert set(row["primary"]) == {"Recall@10", "Prec.@10", "MRR"}
    assert set(row["constraint"]) == {
        "Hallucination-rate (%)",
        "ECE",
        "Brier",
        "Prov. (%)",
    }


def test_verdict_counts_and_seed_summary() -> None:
    counts = verdict_counts({"a": "PASS", "b": "FAIL", "c": "NOT_RUN"})
    assert counts["PASS"] == 1
    assert counts["FAIL"] == 1
    assert counts["NOT_RUN"] == 1
    summary = summarise_seed_runs(
        (
            {
                "recall_at_k": 0.2,
                "mean_conflict": 0.5,
                "mean_interval_width": 0.3,
                "brier": 0.1,
                "width_separability": 0.8,
            },
            {
                "recall_at_k": 0.4,
                "mean_conflict": 0.6,
                "mean_interval_width": 0.4,
                "brier": 0.2,
                "width_separability": 0.9,
            },
        )
    )
    assert summary["runs"] == 2.0
    assert summary["mean_recall"] == pytest.approx(0.3)


def test_ranking_metrics_hand_computed() -> None:
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
    positives = {"a", "d"}
    ranked = rank_order(scores)
    assert recall_at_k(ranked, positives, 2) == pytest.approx(0.5)
    assert precision_at_k(ranked, positives, 2) == pytest.approx(0.5)
    assert mean_reciprocal_rank(ranked, positives) == pytest.approx(1.0)
    assert average_precision(ranked, positives) == pytest.approx((1.0 + 0.5) / 2.0)
    assert hit_rate(ranked, "e", 3) == 0.0
    assert hit_rate(ranked, "d", 4) == 1.0
    curve = recall_curve(scores, positives, (1, 2, 4))
    assert curve[4] == pytest.approx(1.0)
    metrics = evaluate_ranking(scores, positives, k=2)
    assert metrics.positives == 2
    assert fold_averaged_recall((scores, scores), positives, k=2) == pytest.approx(0.5)


def test_ranking_ties_break_deterministically() -> None:
    scores = {"b": 0.5, "a": 0.5, "c": 0.5}
    assert rank_order(scores) == ("a", "b", "c")


def test_calibration_metrics_hand_computed() -> None:
    probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
    labels = np.asarray([0.0, 0.0, 0.0, 1.0])
    assert brier_score(probabilities, labels) == pytest.approx(0.25)
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.25)
    assert overconfidence(probabilities, labels) == pytest.approx(0.25)
    summary = calibration_summary(probabilities, labels, bins=2)
    assert summary.observations == 4
    assert 0.0 <= summary.expected_calibration_error <= 1.0
    curve = reliability_curve(probabilities, labels, bins=2)
    assert sum(entry.count for entry in curve) == 4


def test_calibration_metrics_reject_mismatched_inputs() -> None:
    with pytest.raises(CalibrationMetricError):
        brier_score(np.asarray([0.1, 0.2]), np.asarray([1.0]))


def test_provenance_audit_hand_computed() -> None:
    entry = {
        "candidate": "g",
        "agent": "a",
        "modality": "KG",
        "evidence_nature": "simulated",
        "evidence_score": 0.5,
        "mass_triple": (0.4, 0.3, 0.3),
        "discount_rate": 0.8,
        "calibration_fold": 0,
        "source": "s",
        "source_version": "v",
        "query": "q",
        "recorded_at": "1970-01-01T00:00:00+00:00",
    }
    index = ledger_index((entry,))
    assert ledger_key(entry) == citation_key("g", "a", "s", "v", "q")
    grounded = audit_citations("g", (ledger_key(entry),), index)
    fabricated = audit_citations("g", ("invented",), index)
    assert provenance_grounding_rate((grounded,)) == pytest.approx(1.0)
    assert provenance_grounding_rate((fabricated,)) == pytest.approx(0.0)
    assert hallucinated_edge_rate((fabricated,)) == pytest.approx(1.0)
    partial = audit_citations("g", (ledger_key(entry), "invented"), index)
    assert partial.grounded_rate == pytest.approx(0.5)
    assert unsupported_ledger_fields((entry,)) == {}


def test_provenance_audit_flags_missing_fields() -> None:
    partial = {"candidate": "g", "agent": "a"}
    missing = unsupported_ledger_fields((partial,))  # type: ignore[arg-type]
    assert "source" in missing
    assert missing["source"] == 1


def test_bootstrap_intervals_hand_computed() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    interval = bootstrap_mean_interval(values, resamples=200, seed=1)
    assert interval.point == pytest.approx(3.0)
    assert interval.lower <= interval.upper
    paired = paired_bootstrap_difference(values, values - 1.0, resamples=200, seed=1)
    assert paired.point == pytest.approx(1.0)
    payload = paired_gap_summary(values, values - 0.5, resamples=200, seed=2)
    assert payload["mean_difference"] == pytest.approx(0.5)
    level = seed_level_summary(values, resamples=200, seed=3)
    assert level["seeds"] == 5
    assert level["mean"] == pytest.approx(3.0)


def test_cluster_bootstrap_and_kappa() -> None:
    first = np.asarray([0.7, 0.8, 0.65, 0.9, 0.75, 0.8, 0.7, 0.85])
    second = first - 0.1
    interval = cluster_bootstrap_difference(first, second, resamples=200, seed=4)
    assert interval.point == pytest.approx(0.1)
    agreement = cohens_kappa(np.asarray([1, 0, 1, 0]), np.asarray([1, 0, 1, 1]))
    assert -1.0 <= agreement <= 1.0


def test_cochran_q_hand_computed() -> None:
    effects = np.asarray([0.70, 0.72, 0.69])
    variances = np.asarray([0.0025, 0.0025, 0.0025])
    payload = cochran_q(effects, variances)
    mean = float(np.mean(effects))
    hand = float(np.sum((effects - mean) ** 2 / variances))
    assert payload["q"] == pytest.approx(hand)
    assert payload["pooled"] == pytest.approx(mean)
    assert 0.0 <= payload["i_squared"] <= 1.0
    assert payload["max_gap"] == pytest.approx(0.03)


def test_power_calculation_matches_the_declared_targets() -> None:
    result = prospective_sizing()
    rows = result.as_rows()
    assert [row["analytic_minimum_records"] for row in rows] == [632.0, 542.0, 454.0]
    assert [row["pre_specified_target_records"] for row in rows] == [690.0, 600.0, 500.0]
    assert all(row["target_covers_minimum"] == 1.0 for row in rows)
    assert all(0.80 <= row["power_at_target"] <= 1.0 for row in rows)


def test_accrual_range_covers_every_assumption() -> None:
    payload = accrual_range_covers(prospective_sizing())
    assert payload["all_covered_at_low_end"]
    assert payload["accrual_range"] == [750, 900]


def test_power_rises_with_accrual_and_with_correlation() -> None:
    inputs = SizingInputs()
    powers = [power_at_size(inputs, records, 0.4) for records in (200, 400, 800)]
    assert powers == sorted(powers)
    correlated = [power_at_size(inputs, 600, rho) for rho in (0.2, 0.4, 0.6)]
    assert correlated == sorted(correlated)
    assert normal_cdf(0.0) == pytest.approx(0.5)


def test_sizing_inputs_reject_impossible_requests() -> None:
    with pytest.raises(PowerError):
        SizingInputs(baseline_auroc=1.2).validate()
    with pytest.raises(PowerError):
        SizingInputs(margin=0.5).validate()
    with pytest.raises(PowerError):
        analytic_minimum_records(SizingInputs(), 0.9, ceiling=50)


def test_hanley_mcneil_variance_hand_computed() -> None:
    auroc, positives, negatives = 0.7, 150.0, 450.0
    q1 = auroc / (2.0 - auroc)
    q2 = 2.0 * auroc * auroc / (1.0 + auroc)
    hand = (
        auroc * (1.0 - auroc)
        + (positives - 1.0) * (q1 - auroc * auroc)
        + (negatives - 1.0) * (q2 - auroc * auroc)
    ) / (positives * negatives)
    assert hanley_mcneil_variance(auroc, positives, negatives) == pytest.approx(hand)
    assert paired_variance(0.7, 0.781, positives, negatives, 0.4) < paired_variance(
        0.7, 0.781, positives, negatives, 0.0
    )


def test_delong_auc_equals_the_rank_statistic() -> None:
    generator = np.random.default_rng(12)
    scores = generator.normal(0.7, 1.0, size=200)
    labels = (generator.random(200) < 0.45).astype(int)
    estimate = roc_estimate(scores, labels)
    positives = scores[labels == 1][:, None]
    negatives = scores[labels == 0][None, :]
    hand = float(
        np.mean(
            (positives > negatives).astype(float) + 0.5 * (positives == negatives).astype(float)
        )
    )
    assert estimate.auc == pytest.approx(hand)
    assert estimate.positives == int(labels.sum())


def test_delong_paired_test_is_symmetric() -> None:
    generator = np.random.default_rng(13)
    labels = (generator.random(150) < 0.4).astype(int)
    first = generator.normal(0.5, 1.0, size=150) + labels
    second = generator.normal(0.5, 1.0, size=150)
    forward = paired_comparison(first, second, labels)
    backward = paired_comparison(second, first, labels)
    assert forward.difference == pytest.approx(-backward.difference)
    assert forward.p_value == pytest.approx(backward.p_value)
    assert 0.0 <= forward.p_value <= 1.0


def test_midrank_handles_ties() -> None:
    values = np.asarray([1.0, 2.0, 2.0, 4.0])
    assert midrank(values).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_multiplicity_hand_computed() -> None:
    holm = [entry.adjusted for entry in holm_bonferroni(("a", "b", "c"), [0.01, 0.02, 0.04])]
    assert holm == pytest.approx([0.03, 0.04, 0.04])
    bh = [entry.adjusted for entry in benjamini_hochberg(("a", "b", "c"), [0.01, 0.02, 0.04])]
    assert bh == pytest.approx([0.03, 0.03, 0.04])
    assert holm_bonferroni(("a",), [0.01])[0].rejected
    plan = family_plan()
    assert plan["primary"]["correction"] == "Holm-Bonferroni"
    assert plan["exploratory"]["correction"] == "Benjamini-Hochberg"


def test_multiplicity_is_conservative_relative_to_uncorrected() -> None:
    p_values = [0.001, 0.01, 0.02, 0.2, 0.5]
    names = tuple(f"h{index}" for index in range(len(p_values)))
    assert holm_rejects_only_uncorrected_rejections(names, p_values)
    assert adjusted_never_below_raw(names, p_values)
    adjusted = holm_bonferroni(names, p_values)
    assert [entry.adjusted for entry in adjusted] == pytest.approx([0.005, 0.04, 0.06, 0.4, 0.5])
    assert [entry.rejected for entry in adjusted] == [True, True, False, False, False]


def test_cox_partial_likelihood_matches_the_direct_write_out() -> None:
    times = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    events = np.asarray([1, 1, 0, 1, 0, 1])
    covariates = np.asarray([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    fitted = fit_cox(times, events, covariates, names=("arm",))
    beta = float(fitted.coefficients[0])
    hand = 0.0
    for index in range(times.size):
        if events[index] == 0:
            continue
        at_risk = times >= times[index]
        hand += beta * covariates[index, 0] - np.log(
            float(np.sum(np.exp(beta * covariates[at_risk, 0])))
        )
    assert fitted.log_likelihood == pytest.approx(hand)
    assert 0.0 <= fitted.concordance <= 1.0
    assert fitted.events == 4
    assert len(fitted.as_rows()) == 1
    assert fitted.confidence_intervals()[0][0] <= fitted.coefficients[0]


def test_cox_rejects_a_table_without_events() -> None:
    with pytest.raises(SurvivalError):
        fit_cox(
            np.asarray([1.0, 2.0]),
            np.asarray([0, 0]),
            np.asarray([[0.0], [1.0]]),
            names=("arm",),
        )


def test_stratified_cox_and_competing_risks() -> None:
    generator = np.random.default_rng(21)
    times = generator.uniform(1.0, 20.0, size=60)
    covariates = generator.normal(size=(60, 1))
    event_type = (generator.random(60) < 0.4).astype(int)
    event_type[::5] = 2
    strata = np.repeat([0, 1, 2], 20)
    fit = fit_cox(times, event_type > 0, covariates, names=("x",), strata=strata)
    assert fit.events > 0
    competing = fit_competing_risks(times, event_type, covariates, names=("x",), strata=strata)
    assert competing.competing_events > 0
    incidence = cumulative_incidence(times, event_type)
    assert incidence.size == np.unique(times).size
    assert np.all(np.diff(incidence) >= -1e-12)
    assert incidence[-1] <= 1.0 + 1e-9
    gap = site_gap_summary((0.70, 0.76, 0.72))
    assert gap["gap"] == pytest.approx(0.06)
    assert gap["flagged"] == 0.0


def test_concordance_index_hand_computed() -> None:
    times = np.asarray([1.0, 2.0, 3.0])
    events = np.asarray([1, 1, 1])
    risk = np.asarray([3.0, 2.0, 1.0])
    assert concordance_index(times, events, risk) == pytest.approx(1.0)
    assert concordance_index(times, events, -risk) == pytest.approx(0.0)


def test_reader_study_recovers_the_generating_effect() -> None:
    generator = np.random.default_rng(31)
    readers = np.repeat(np.arange(9), 40)
    arms = np.tile(np.repeat([0, 1], 20), 9)
    linear = -0.4 + 0.7 * arms
    per_case = (generator.random(readers.size) < 1.0 / (1.0 + np.exp(-linear))).astype(float)
    design, names = reader_study_design(per_case, readers, arms)
    result = fit_gee_logistic(per_case, design, readers, names=names)
    report = arm_gain_report(result)
    assert report["readers"] == 9
    assert report["observations"] == readers.size
    assert abs(report["coefficient"] - 0.7) < 0.9
    assert report["ci_lower"] <= report["coefficient"] <= report["ci_upper"]
    assert 0.0 <= report["p_value"] <= 1.0
    assert 0.0 <= result.working_correlation <= 0.95


def test_reader_study_rejects_a_single_reader() -> None:
    with pytest.raises(GeeError):
        fit_gee_logistic(
            np.asarray([1.0, 0.0, 1.0, 0.0]),
            np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
            np.asarray([1, 1, 1, 1]),
            names=("intercept", "arm"),
        )


def test_simulation_design_reports_its_saturated_cutoff() -> None:
    design = SimulationDesign()
    assert design.saturated_cutoff == 45
    assert design.total_runs == 180
    single = SimulationDesign(cells=(SimulationCell(0.3, 0.0, seeds=(1,), n_candidates=100),))
    assert single.saturated_cutoff == 10
