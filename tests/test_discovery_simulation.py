"""Discovery pass, provenance ledger, simulation study and the ablation battery."""

from __future__ import annotations

import json

import numpy as np
import pytest

from crb_concordance.agents.base import AgentContext, RawEvidence
from crb_concordance.agents.mapping import MassMapper
from crb_concordance.agents.verifier import VerificationError, Verifier
from crb_concordance.discovery.ledger import ProvenanceLedger
from crb_concordance.discovery.pipeline import DiscoveryConfig, DiscoveryError, run_discovery
from crb_concordance.discovery.ranking import (
    rank_reports,
    score_map,
    separability,
    top_names,
)
from crb_concordance.evaluation.ablations import (
    ExtractionLevel,
    SubstitutionRule,
    full_plan,
    run_battery,
    score_pool,
)
from crb_concordance.metrics.provenance import REQUIRED_LEDGER_FIELDS, summarise
from crb_concordance.metrics.ranking import rank_of
from crb_concordance.simulation.contrasts import (
    all_cell_contrasts,
    conflict_monotonicity_across_cells,
    conflict_trace_h2,
    cutoff_reachability,
    falsification_inequality,
    least_squares_r2,
    partial_r2,
    pooled_trace_analysis,
)
from crb_concordance.simulation.design import (
    ABSENT_GRID,
    CONFLICT_GRID,
    SIMULATED_CANDIDATES,
    SIMULATED_PREVALENCE,
    SIMULATION_SEEDS,
    SimulationCell,
    SimulationDesign,
    default_design,
)
from crb_concordance.simulation.evidence_model import (
    absent_only_width,
    best_single_modality,
    summarise_pool,
    summarise_pool_naive,
)
from crb_concordance.simulation.runner import run_design, width_separability
from crb_concordance.utils.types import MODALITY_ORDER, Modality


@pytest.fixture(scope="module")
def runtime(request):
    from crb_concordance.calibration.discount_fit import calibrate
    from crb_concordance.cohorts.control_panel import build_control_panel
    from crb_concordance.cohorts.generator import (
        CohortDesign,
        generate_cohort,
        generate_expression,
    )
    from crb_concordance.graph.generator import build_substrate, dependency_table_from_substrate
    from crb_concordance.graph.paths import PathFinder
    from crb_concordance.metabolism.model import StoichiometricModel

    bundle = build_substrate()
    finder = PathFinder(bundle.graph, max_hops=2)
    records = generate_cohort(CohortDesign())
    genes = tuple(bundle.candidate_pool[:16])
    profiles = generate_expression(records, genes)
    panel = build_control_panel()
    context = AgentContext(
        finder=finder,
        phenotype=bundle.phenotype,
        model=StoichiometricModel.from_catalogue(),
        dependency_table=dependency_table_from_substrate(bundle),
        profiles=profiles,
        records=records,
        panel=panel,
    )
    return {
        "context": context,
        "rates": calibrate(panel).pooled,
        "genes": genes,
        "bundle": bundle,
        "panel": panel,
    }


def test_discovery_pools_every_candidate_with_structural_invariants(runtime) -> None:
    ledger = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    result = run_discovery(runtime["genes"], runtime["context"], runtime["rates"], ledger=ledger)
    assert len(result.ranked) == len(runtime["genes"])
    assert result.structural["all_triples_normalised"]
    assert result.structural["all_pignistic_inside_interval"]
    assert len(ledger.entries) == len(runtime["genes"]) * len(MODALITY_ORDER)
    assert all(set(REQUIRED_LEDGER_FIELDS) <= set(row) for row in ledger.entries)


def test_discovery_ledger_rows_carry_masses_that_sum_to_one(runtime) -> None:
    ledger = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    run_discovery(runtime["genes"], runtime["context"], runtime["rates"], ledger=ledger)
    for row in ledger.entries:
        assert sum(row["mass_triple"]) == pytest.approx(1.0, abs=1e-7)
        assert 0.0 <= row["discount_rate"] <= 1.0
        assert row["agent"]
        assert row["evidence_nature"]


def test_ledger_flush_appends_without_rewriting(tmp_path, runtime) -> None:
    first = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    run_discovery(runtime["genes"][:2], runtime["context"], runtime["rates"], ledger=first)
    target = tmp_path / "ledger.jsonl"
    first.flush(target)
    initial = target.read_text(encoding="utf-8").count("\n")
    second = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    run_discovery(runtime["genes"][:2], runtime["context"], runtime["rates"], ledger=second)
    second.flush(target)
    appended = target.read_text(encoding="utf-8").count("\n")
    assert appended == 2 * initial
    assert json.loads(target.read_text(encoding="utf-8").splitlines()[0])["candidate"]


def test_provenance_grounding_covers_every_citation(runtime) -> None:
    ledger = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    result = run_discovery(runtime["genes"], runtime["context"], runtime["rates"], ledger=ledger)
    payload = result.provenance()
    assert payload["provenance_grounding_rate"] == pytest.approx(1.0)
    assert payload["hallucinated_edge_rate"] == pytest.approx(0.0)
    assert payload["complete_entries"] == len(ledger)


def test_empty_pool_is_rejected(runtime) -> None:
    with pytest.raises(DiscoveryError):
        run_discovery((), runtime["context"], runtime["rates"])


def test_ranking_is_monotone_and_deterministic(runtime) -> None:
    ledger = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    result = run_discovery(runtime["genes"], runtime["context"], runtime["rates"], ledger=ledger)
    scores = [entry.pignistic for entry in result.ranked]
    assert scores == sorted(scores, reverse=True)
    assert top_names(result.ranked, 3) == tuple(entry.candidate for entry in result.ranked[:3])
    assert rank_of(result.scores(), result.ranked[0].candidate) == 1
    assert set(score_map(result.ranked)) == set(runtime["genes"])


def test_separability_separates_narrow_from_wide(runtime) -> None:
    ledger = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    result = run_discovery(runtime["genes"], runtime["context"], runtime["rates"], ledger=ledger)
    payload = separability(result.ranked, width_ceiling=0.3)
    assert 0.0 <= payload["auroc"] <= 1.0
    assert payload["concordant"] + payload["discordant"] == len(result.ranked)


def test_verifier_rejects_an_incomplete_evidence_set(runtime, learned_rates) -> None:
    verifier = Verifier(rates=runtime["rates"], mapper=MassMapper())
    evidences = {Modality.KG: RawEvidence("g", Modality.KG, 0.5, "simulated")}
    with pytest.raises(VerificationError):
        verifier.combine_candidate("g", evidences)


def test_verifier_keeps_absent_evidence_vacuous(runtime) -> None:
    verifier = Verifier(rates=runtime["rates"], mapper=MassMapper())
    evidences = {
        modality: RawEvidence("g", modality, None, "absent") for modality in MODALITY_ORDER
    }
    outcome = verifier.combine_candidate("g", evidences)
    assert outcome.report.interval_width == pytest.approx(1.0)
    assert outcome.report.pignistic == pytest.approx(0.5)
    assert outcome.combined.total_conflict == pytest.approx(0.0)


def test_default_simulation_design_matches_the_declared_grid() -> None:
    design = default_design()
    assert len(design.cells) == 9
    assert all(len(cell.seeds) == 20 for cell in design.cells)
    assert all(cell.n_candidates == 2000 for cell in design.cells)
    assert all(cell.prevalence == 0.15 for cell in design.cells)
    assert {cell.conflict_probability for cell in design.cells} == set(CONFLICT_GRID)
    assert {cell.absent_fraction for cell in design.cells} == set(ABSENT_GRID)
    assert tuple(range(1, 21)) == SIMULATION_SEEDS
    assert SIMULATED_CANDIDATES == 2000
    assert SIMULATED_PREVALENCE == 0.15
    assert design.bootstrap_resamples == 5000


def test_simulation_runs_are_reproducible(learned_rates) -> None:
    cells = (
        SimulationCell(0.3, 0.0, seeds=(1, 2), n_candidates=150),
        SimulationCell(0.5, 0.25, seeds=(1, 2), n_candidates=150),
    )
    design = SimulationDesign(cells=cells, bootstrap_resamples=100)
    first = run_design(learned_rates, design=design)
    second = run_design(learned_rates, design=design)
    assert [run.as_dict() for run in first.runs()] == [run.as_dict() for run in second.runs()]


def test_absent_evidence_widens_the_intervals(learned_rates) -> None:
    narrower = summarise_pool(
        SimulationCell(0.3, 0.0, seeds=(1,), n_candidates=400).candidates(1), learned_rates
    )
    wider = summarise_pool(
        SimulationCell(0.3, 0.5, seeds=(1,), n_candidates=400).candidates(1), learned_rates
    )
    assert float(np.mean([summary.interval_width for summary in wider])) > float(
        np.mean([summary.interval_width for summary in narrower])
    )


def test_fully_absent_candidates_keep_unit_ignorance(learned_rates) -> None:
    summaries = summarise_pool(
        SimulationCell(0.3, 0.5, seeds=(1,), n_candidates=400).candidates(1), learned_rates
    )
    assert absent_only_width(summaries) in (0.0, 1.0)
    absent = [s for s in summaries if s.absent_modalities == len(MODALITY_ORDER)]
    assert all(summary.interval_width == pytest.approx(1.0) for summary in absent)


def test_best_single_modality_is_chosen_from_the_pool(learned_rates) -> None:
    summaries = summarise_pool(
        SimulationCell(0.3, 0.0, seeds=(1,), n_candidates=300).candidates(1), learned_rates
    )
    modality, recall = best_single_modality(summaries, cutoff=10)
    assert modality in MODALITY_ORDER
    assert 0.0 <= recall <= 1.0


def test_partial_r_squared_is_zero_for_identical_designs() -> None:
    generator = np.random.default_rng(2)
    target = generator.normal(size=60)
    design = generator.normal(size=(60, 2))
    assert partial_r2(design, design, target) == pytest.approx(0.0)
    assert 0.0 <= least_squares_r2(design, target) <= 1.0


def test_conflict_trace_h2_adds_variance(learned_rates) -> None:
    summaries = summarise_pool(
        SimulationCell(0.3, 0.25, seeds=(1,), n_candidates=400).candidates(1),
        learned_rates,
        conflict_threshold=0.05,
    )
    outcome, added, full, reduced = conflict_trace_h2(summaries)
    assert full >= reduced
    assert added >= 0.0
    assert outcome.name == "H2_conflict_trace_partial_r2"


def test_conflict_rises_with_the_declared_conflict_probability(learned_rates) -> None:
    cells = tuple(
        SimulationCell(level, 0.0, seeds=(1, 2), n_candidates=300) for level in (0.1, 0.3, 0.5)
    )
    result = run_design(learned_rates, design=SimulationDesign(cells=cells, bootstrap_resamples=50))
    outcome = conflict_monotonicity_across_cells(result)
    assert outcome.statistic > 0.0


def test_cutoff_reachability_reports_the_bound(learned_rates) -> None:
    cells = (SimulationCell(0.3, 0.0, seeds=(1,), n_candidates=2000),)
    result = run_design(learned_rates, design=SimulationDesign(cells=cells, bootstrap_resamples=50))
    payload = cutoff_reachability(result)
    assert payload["recall_ceiling_at_declared_cutoff"] == pytest.approx(10 / 300)
    assert payload["margin_reachable_at_declared_cutoff"] == 0.0
    assert payload["first_reachable_cutoff"] == 45.0


def test_falsification_inequality_reports_a_verdict(learned_rates) -> None:
    cells = (SimulationCell(0.3, 0.0, seeds=(1, 2), n_candidates=300),)
    result = run_design(learned_rates, design=SimulationDesign(cells=cells, bootstrap_resamples=50))
    outcome = falsification_inequality(result)
    assert outcome.verdict.value in {"PASS", "FAIL"}
    assert outcome.threshold == pytest.approx(0.15)


def test_width_separability_uses_an_external_label(learned_rates) -> None:
    summaries = summarise_pool(
        SimulationCell(0.3, 0.25, seeds=(1,), n_candidates=400).candidates(1), learned_rates
    )
    value = width_separability(summaries, margin=0.25)
    assert 0.0 <= value <= 1.0


def test_pooled_trace_analysis_covers_h1_and_h2(learned_rates) -> None:
    cells = (SimulationCell(0.3, 0.25, seeds=(1, 2), n_candidates=200),)
    payload = pooled_trace_analysis(
        learned_rates, SimulationDesign(cells=cells, bootstrap_resamples=50), seeds_per_cell=1
    )
    assert payload["candidates"] > 0
    assert "H1" in payload["h1"]["name"]
    assert "H2" in payload["h2"]["name"]
    assert set(payload["single_modality_recall"]) == {"KG", "DEP", "FLUX", "CLIN"}


def test_naive_fusion_scores_the_same_pool(learned_rates) -> None:
    candidates = SimulationCell(0.3, 0.0, seeds=(1,), n_candidates=200).candidates(1)
    full = summarise_pool(candidates, learned_rates)
    naive = summarise_pool_naive(candidates, learned_rates)
    assert {s.gene for s in full} == {s.gene for s in naive}
    assert all(summary.total_conflict == 0.0 for summary in naive)
    assert any(
        full[index].pignistic != pytest.approx(naive[index].pignistic) for index in range(len(full))
    )


def test_ablation_plan_covers_every_level_and_rule() -> None:
    plan = full_plan()
    assert set(plan["extraction_levels"]) == {level.value for level in ExtractionLevel}
    assert set(plan["per_modality_removals"]) == {modality.value for modality in MODALITY_ORDER}
    assert set(plan["substitution_rules"]) == {rule.value for rule in SubstitutionRule}


def test_ablation_battery_reports_every_cell(learned_rates, small_pool) -> None:
    outcomes = run_battery(small_pool, learned_rates, cutoff=20)
    labels = {outcome.label for outcome in outcomes}
    assert "full" in labels
    assert "no-knowledge-graph" in labels
    assert any(label.startswith("remove ") for label in labels)
    assert any(label.startswith("single-agent ") for label in labels)
    assert any(label.startswith("substitution: ") for label in labels)
    assert all(0.0 <= outcome.mean_interval_width <= 1.0 for outcome in outcomes)


def test_removing_the_knowledge_graph_changes_the_interval(learned_rates, small_pool) -> None:
    full = score_pool(small_pool, learned_rates)
    without = score_pool(small_pool, learned_rates, level=ExtractionLevel.NO_KNOWLEDGE_GRAPH)
    assert full.widths != without.widths


def test_dempster_substitution_collapses_the_interval(learned_rates, small_pool) -> None:
    full = score_pool(small_pool, learned_rates)
    dempster = score_pool(small_pool, learned_rates, rule=SubstitutionRule.DEMPSTER_RENORMALISED)
    full_width = float(np.mean(list(full.widths.values())))
    dempster_width = float(np.mean(list(dempster.widths.values())))
    assert dempster_width < full_width


def test_rank_reports_orders_by_pignistic(runtime) -> None:
    ledger = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    result = run_discovery(
        runtime["genes"][:5], runtime["context"], runtime["rates"], ledger=ledger
    )
    reordered = rank_reports(tuple(reversed(tuple(entry.report for entry in result.ranked))))
    assert [entry.rank for entry in reordered] == [1, 2, 3, 4, 5]


def test_discovery_config_records_its_settings(runtime) -> None:
    config = DiscoveryConfig(conflict_threshold=0.1, max_hops=2)
    payload = config.as_dict()
    assert payload["conflict_threshold"] == 0.1
    assert payload["max_hops"] == 2
    ledger = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    result = run_discovery(
        runtime["genes"][:2], runtime["context"], runtime["rates"], config=config, ledger=ledger
    )
    assert result.config.conflict_threshold == 0.1


def test_cell_contrasts_are_seed_level_summaries(learned_rates) -> None:
    cells = (SimulationCell(0.3, 0.0, seeds=(1, 2, 3), n_candidates=200),)
    result = run_design(
        learned_rates, design=SimulationDesign(cells=cells, bootstrap_resamples=100)
    )
    contrasts = all_cell_contrasts(result)
    assert len(contrasts) == 1
    payload = contrasts[0].as_dict()
    assert payload["recall"]["seeds"] == 3
    assert (
        payload["recall"]["ci_lower"] <= payload["recall"]["mean"] <= payload["recall"]["ci_upper"]
    )


def test_summarise_returns_a_provenance_summary(runtime) -> None:
    ledger = ProvenanceLedger(clock=lambda: "1970-01-01T00:00:00+00:00")
    result = run_discovery(
        runtime["genes"][:3], runtime["context"], runtime["rates"], ledger=ledger
    )
    summary = summarise(result.audits, tuple(ledger.entries))
    assert summary.candidates == 3
    assert 0.0 <= summary.grounded_rate <= 1.0
