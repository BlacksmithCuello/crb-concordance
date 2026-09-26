"""Minimal training closed loop and the end-to-end smoke path on the smoke config."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from crb_concordance.agents.mapping import MassMapper
from crb_concordance.calibration.mass_calibration import (
    CalibratedMassMap,
    batches_from_table,
    brier_mass_loss,
    build_table,
)
from crb_concordance.cohorts.control_panel import build_control_panel
from crb_concordance.simulation.design import SimulationCell, SimulationDesign
from crb_concordance.simulation.runner import run_design
from crb_concordance.training.engine import EngineError, evaluate_loss, fit
from crb_concordance.training.optim import OptimConfig, ScheduleError, learning_rate_at
from crb_concordance.utils.config import (
    ConfigError,
    apply_overrides,
    coerce_scalar,
    load_experiment,
)


def test_smoke_config_declares_reduced_sizes(config_root) -> None:
    config = load_experiment(config_root, "_smoke")
    assert config.get("simulation.candidates_per_seed") == 120
    assert config.get("simulation.seeds") == 2
    assert config.get("data.cohort_size") == 3000
    assert config.get("model.max_hops") == 2


def test_main_config_matches_the_declared_design(config_root) -> None:
    config = load_experiment(config_root, "main")
    assert config.get("simulation.seeds") == 20
    assert config.get("simulation.candidates_per_seed") == 2000
    assert config.get("simulation.prevalence") == 0.15
    assert config.get("simulation.conflict_probability_grid") == [0.1, 0.3, 0.5]
    assert config.get("simulation.absent_evidence_fraction_grid") == [0.0, 0.25, 0.5]
    assert config.get("simulation.seed_level_bootstrap_resamples") == 5000


def test_config_overrides_are_applied(config_root) -> None:
    config = load_experiment(
        config_root, "main", overrides=["model.max_hops=2", "simulation.seeds=3"]
    )
    assert config.get("model.max_hops") == 2
    assert config.get("simulation.seeds") == 3


def test_config_scalar_coercion_and_merging() -> None:
    assert coerce_scalar("true") is True
    assert coerce_scalar("12") == 12
    assert coerce_scalar("0.25") == 0.25
    assert coerce_scalar("none") is None
    assert coerce_scalar("[1, 2.5, x]") == [1, 2.5, "x"]
    merged = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c=3"])
    assert merged["a"] == {"b": 2, "c": 3}
    with pytest.raises(ConfigError):
        apply_overrides({}, ["malformed"])


def test_two_step_training_reduces_the_loss() -> None:
    panel = build_control_panel()
    table = build_table(panel)
    batches = batches_from_table(table, 5)
    module = CalibratedMassMap()
    initial = evaluate_loss(module, batches, brier_mass_loss)
    history = fit(
        module,
        batches,
        brier_mass_loss,
        OptimConfig(
            learning_rate=0.08,
            epochs=2,
            batch_size=5,
            warmup_fraction=0.5,
            seed=1,
            grad_accum=2,
        ),
        ema=False,
    )
    assert len(history.records) == 2 * len(batches)
    assert history.final_loss < initial
    assert history.decreased()


def test_gradients_flow_to_every_parameter() -> None:
    table = build_table(build_control_panel())
    batches = batches_from_table(table, table.observations)
    module = CalibratedMassMap()
    loss = brier_mass_loss(module, *batches[0])
    loss.backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_mass_map_has_exactly_the_declared_parameters() -> None:
    module = CalibratedMassMap()
    names = {name for name, _ in module.named_parameters()}
    assert names == {"slope", "offset", "gate_logit"}
    for parameter in module.parameters():
        assert parameter.shape == (4,)


def test_engine_rejects_empty_batches() -> None:
    with pytest.raises(EngineError):
        fit(CalibratedMassMap(), [], brier_mass_loss, OptimConfig())


def test_schedule_configuration_is_validated() -> None:
    with pytest.raises(ScheduleError):
        OptimConfig(learning_rate=0.0).validate()
    with pytest.raises(ScheduleError):
        OptimConfig(scheduler="triangular").validate()
    with pytest.raises(ScheduleError):
        OptimConfig(precision="fp8").validate()
    with pytest.raises(ScheduleError):
        OptimConfig(warmup_fraction=1.5).validate()
    with pytest.raises(ScheduleError):
        learning_rate_at(-1, 10, OptimConfig())


def test_effective_batch_matches_the_declared_composition() -> None:
    config = OptimConfig(batch_size=8, grad_accum=4)
    assert config.effective_batch == 32
    assert config.as_dict()["effective_batch"] == 32


def test_checkpoint_payload_is_content_hashed(tmp_path) -> None:
    module = CalibratedMassMap()
    batches = batches_from_table(build_table(build_control_panel()), 5)
    history = fit(
        module,
        batches,
        brier_mass_loss,
        OptimConfig(epochs=2, batch_size=5, seed=11),
        checkpoint_path=tmp_path / "a.ckpt",
        ema=False,
    )
    assert history.checkpoint is not None
    first = json.dumps(history.checkpoint.as_dict(), sort_keys=True)
    history_repeat = fit(
        CalibratedMassMap(),
        batches,
        brier_mass_loss,
        OptimConfig(epochs=2, batch_size=5, seed=11),
        checkpoint_path=tmp_path / "b.ckpt",
        ema=False,
    )
    assert history_repeat.checkpoint is not None
    assert json.dumps(history_repeat.checkpoint.as_dict(), sort_keys=True) == first


def test_ema_snapshot_is_taken_when_requested(tmp_path) -> None:
    panel = build_control_panel()
    batches = batches_from_table(build_table(panel), 5)
    module = CalibratedMassMap()
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    fit(
        module,
        batches,
        brier_mass_loss,
        OptimConfig(epochs=30, batch_size=5, seed=2, ema_decay=0.5, learning_rate=0.05),
        checkpoint_path=tmp_path / "ema.ckpt",
        ema=True,
    )
    moved = any(
        float((module.state_dict()[name] - before[name]).abs().max()) > 0.0 for name in before
    )
    assert moved


def test_full_smoke_loop_runs_on_the_smoke_config(learned_rates) -> None:
    """A minimal training-loop analogue for the simulation: run, score, rank, report."""

    cells = (
        SimulationCell(0.3, 0.0, seeds=(1, 2), n_candidates=120),
        SimulationCell(0.5, 0.25, seeds=(1, 2), n_candidates=120),
    )
    design = SimulationDesign(cells=cells, bootstrap_resamples=200)
    result = run_design(learned_rates, design=design)
    runs = result.runs()
    assert len(runs) == 4
    recalls = np.asarray([run.recall_at_k for run in runs], dtype=float)
    assert np.all(recalls >= 0.0)
    assert np.all(recalls <= 1.0)
    assert all(run.mean_interval_width > 0.0 for run in runs)
    assert all(0.0 <= run.brier <= 1.0 for run in runs)


def test_mapper_produces_monotone_pignistic_scores() -> None:
    mapper = MassMapper()
    from crb_concordance.utils.types import Modality

    lows = [
        mapper.map_score(Modality.DEP, value).pignistic() for value in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    assert lows == sorted(lows)
    assert lows[0] < lows[-1]
