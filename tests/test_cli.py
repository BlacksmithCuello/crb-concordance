"""Command-line entry points, exercised end to end on the smoke configuration."""

from __future__ import annotations

import json

import pytest

from crb_concordance.cli.ablate import run as run_ablate
from crb_concordance.cli.benchmark import run as run_benchmark
from crb_concordance.cli.calibrate import run as run_calibrate
from crb_concordance.cli.discover import run as run_discover
from crb_concordance.cli.runtime import StudyRuntime, build_runtime
from crb_concordance.cli.simulate import build_design
from crb_concordance.cli.simulate import run as run_simulate
from crb_concordance.cli.size_prospective import run as run_sizing
from crb_concordance.simulation.design import SimulationDesign
from crb_concordance.utils.config import load_experiment
from tests.conftest import CONFIG_ROOT


@pytest.fixture(scope="module")
def runtime(config_root) -> StudyRuntime:
    return build_runtime(config_root=config_root, experiment="_smoke")


def test_runtime_assembles_every_component(runtime) -> None:
    assert runtime.candidate_pool()
    assert runtime.rates.as_dict()
    assert runtime.panel.symbols()
    assert runtime.records
    assert runtime.model.n_reactions > 0
    assert runtime.notes["calibration_rates"]


def test_calibration_cli_writes_its_reports(runtime, config_root, tmp_path) -> None:
    payload = run_calibrate(
        config_root=config_root, experiment="_smoke", overrides=[], output_root=tmp_path
    )
    directory = tmp_path / "_smoke"
    assert (directory / "calibration_report.json").exists()
    assert (directory / "calibration_report.txt").exists()
    assert (directory / "mass_map.ckpt").exists()
    assert payload["mass_map"]["steps"] > 0
    assert payload["discount_rates"]["pooled"]


def test_discovery_cli_writes_the_ranking_and_ledger(runtime, config_root, tmp_path) -> None:
    payload = run_discover(
        config_root=config_root,
        experiment="_smoke",
        overrides=[],
        output_root=tmp_path,
        pool_limit=6,
    )
    directory = tmp_path / "_smoke"
    assert (directory / "discovery_report.json").exists()
    ledger = directory / "provenance_ledger.jsonl"
    assert ledger.exists()
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 6 * 4
    assert payload["structural"]["all_triples_normalised"]
    assert payload["provenance"]["provenance_grounding_rate"] == pytest.approx(1.0)
    assert payload["separability"]["auroc"] >= 0.0


def test_simulation_cli_runs_a_reduced_grid(config_root, tmp_path) -> None:
    payload = run_simulate(
        config_root=config_root,
        experiment="_smoke",
        overrides=[],
        output_root=tmp_path,
        seeds=1,
        candidates=80,
        bootstrap=100,
        pooled_seeds=1,
    )
    directory = tmp_path / "_smoke"
    assert (directory / "simulation_report.json").exists()
    assert (directory / "simulation_report.txt").exists()
    # the smoke configuration declares a single grid cell
    assert payload["design"]["total_runs"] == 1
    assert payload["design"]["cells"][0]["n_candidates"] == 80
    assert payload["hypotheses"]["outcomes"]
    assert payload["pooled_trace_analysis"]["candidates"] > 0


def test_ablation_cli_covers_the_battery(config_root, tmp_path) -> None:
    payload = run_ablate(experiment="_smoke", output_root=tmp_path, n_candidates=120, cutoff=6)
    labels = {outcome["label"] for outcome in payload["outcomes"]}
    assert "full" in labels
    assert "no-verifier (naive average)" in labels
    assert any(label.startswith("substitution: ") for label in labels)
    assert payload["cutoff"] == 6
    assert (tmp_path / "_smoke" / "ablation_report.txt").exists()


def test_sizing_cli_reproduces_the_declared_targets(config_root, tmp_path) -> None:
    payload = run_sizing(
        experiment="supplementary_prospective_sizing",
        output_root=tmp_path,
        baseline_auroc=0.70,
        margin=0.081,
        prevalence=0.25,
        alpha=0.05,
        power=0.80,
    )
    assert [row["pre_specified_target_records"] for row in payload["rows"]] == [690.0, 600.0, 500.0]
    assert payload["accrual_coverage"]["all_covered_at_low_end"]
    assert (tmp_path / "supplementary_prospective_sizing" / "sizing_report.json").exists()


def test_benchmark_cli_binds_every_row(config_root, tmp_path) -> None:
    payload = run_benchmark(
        config_root=config_root, experiment="_smoke", overrides=[], output_root=tmp_path, seed=1
    )
    assert payload["registry"]["baselines"] == 37
    assert payload["registry"]["fusion_controls"] == 2
    assert len(payload["substrate_pool_ranking"]) == 39
    assert payload["table_one"]["rows"][-1]["method"].startswith("CRB-Concordance")
    assert all(not row["executed"] for row in payload["table_one"]["rows"])
    assert "control_ranks" in payload
    assert (tmp_path / "_smoke" / "benchmark_report.json").exists()


def test_report_renderer_never_emits_markdown(tmp_path, config_root) -> None:
    run_ablate(experiment="ablation_no_verifier", output_root=tmp_path, n_candidates=60, cutoff=5)
    text = (tmp_path / "ablation_no_verifier" / "ablation_report.txt").read_text(encoding="utf-8")
    assert not any(line.lstrip().startswith("#") for line in text.splitlines())
    assert "|" not in text
    assert text.endswith("\n")


def test_build_design_respects_reductions() -> None:
    reduced = build_design(seeds=2, candidates=100, bootstrap=50)
    assert all(len(cell.seeds) == 2 for cell in reduced.cells)
    assert all(cell.n_candidates == 100 for cell in reduced.cells)
    assert reduced.bootstrap_resamples == 50
    assert isinstance(reduced, SimulationDesign)
    assert build_design(seeds=None, candidates=None, bootstrap=None).total_runs == 180

    config = load_experiment(CONFIG_ROOT, "main")
    from_config = build_design(seeds=None, candidates=None, bootstrap=None, config=config)
    assert from_config.total_runs == 180
    assert from_config.saturated_cutoff == 45
