"""Cohort design, control panel and the calibration path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crb_concordance.calibration.discount_fit import (
    calibrate,
    closed_form_agreement,
    fit_rate_closed_form,
    fit_rate_numeric,
    rate_loss,
    single_modality_pignistic,
)
from crb_concordance.calibration.folds import (
    FoldPolicy,
    audit_folds,
    build_folds,
    fold_for_symbol,
    withheld_positives,
)
from crb_concordance.calibration.mass_calibration import (
    CalibratedMassMap,
    batches_from_table,
    brier_mass_loss,
    build_table,
    fit_mass_map,
)
from crb_concordance.cohorts.control_panel import (
    CALIBRATION_GENES,
    CONTROL_GENES,
    RankExpectation,
    build_control_panel,
    modality_observations,
)
from crb_concordance.cohorts.generator import (
    CohortDesign,
    EvidenceDesign,
    generate_expression,
    simulate_candidates,
)
from crb_concordance.cohorts.schema import (
    SITE_REGION,
    Arm,
    SchemaError,
    census,
    labelled_records,
    retained_records,
)
from crb_concordance.training.checkpointing import CheckpointError, load_checkpoint, resume_seed
from crb_concordance.training.engine import evaluate_loss, fit
from crb_concordance.training.optim import (
    OptimConfig,
    learning_rate_at,
    schedule_curve,
    total_steps,
)
from crb_concordance.utils.types import MODALITY_ORDER, Modality


def test_cohort_design_matches_the_reported_ranges() -> None:
    design = CohortDesign()
    design.validate()
    assert 3000 <= design.retrospective_size <= 3600
    assert all(1000 <= count <= 1200 for count in design.retrospective_site_counts)
    assert 750 <= design.prospective_size <= 900
    assert 0.20 <= design.pcr_rate <= 0.30
    assert 0.65 <= design.soc_auroc <= 0.74
    assert design.reader_count >= 8
    assert design.shared_cases >= 155


def test_cohort_design_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        CohortDesign(retrospective_site_counts=(10, 10, 10)).validate()
    with pytest.raises(ValueError):
        CohortDesign(pcr_rate=0.5).validate()
    with pytest.raises(ValueError):
        CohortDesign(soc_auroc=0.9).validate()


def test_generated_cohort_matches_the_design_marginals(cohort_records) -> None:
    report = census(cohort_records)
    assert report.total == 3300 + 825
    assert report.by_arm["retrospective"] == 3300
    assert report.by_arm["prospective"] == 825
    assert set(report.by_site) == {"Site A", "Site B", "Site C"}
    assert len(report.by_region) == 3
    assert 0.20 <= report.pcr_rate <= 0.30
    assert report.exclusions
    assert report.retained == report.total - report.excluded
    assert report.excluded >= sum(report.exclusions.values())
    assert all(record.retained() for record in retained_records(cohort_records))


def test_record_validation_rejects_a_mismatched_site(cohort_records) -> None:
    record = cohort_records[0]
    wrong_region = "Region III" if record.region != "Region III" else "Region I"
    with pytest.raises(SchemaError):
        replace(record, region=wrong_region).validate()
    with pytest.raises(SchemaError):
        replace(record, ct_stage=9).validate()
    with pytest.raises(SchemaError):
        replace(record, site="Site Z").validate()


def test_labelled_records_are_a_subset_of_retained(cohort_records) -> None:
    retained = set(record.record_id for record in retained_records(cohort_records))
    labelled = labelled_records(cohort_records)
    assert labelled
    assert all(record.record_id in retained for record in labelled)
    assert all(record.binary_label() is not None for record in labelled)


def test_site_region_mapping_is_consistent(cohort_records) -> None:
    for record in cohort_records:
        assert record.region == SITE_REGION[record.site]
        assert record.arm in (Arm.RETROSPECTIVE, Arm.PROSPECTIVE)


def test_expression_stand_in_covers_the_retained_labelled_records(cohort_records) -> None:
    genes = ("SLC2A1", "LDHA", "G6PD")
    profiles = generate_expression(cohort_records, genes)
    labelled = labelled_records(cohort_records)
    assert len(profiles) == len(labelled)
    sample = next(iter(profiles.values()))
    assert sample.profile().keys() == set(genes)
    assert sample.gene("SLC2A1") is not None
    assert sample.gene("ABSENT") is None


def test_control_panel_declares_roles_and_expectations() -> None:
    panel = build_control_panel()
    assert [gene.symbol for gene in CONTROL_GENES] == [
        "SLC2A1",
        "SLC16A1",
        "SLC16A9",
        "CPT1A",
        "CES1",
    ]
    assert panel.positives() == ("SLC2A1", "SLC16A1")
    assert panel.negatives() == ("SLC16A9", "CPT1A", "CES1")
    expectations = panel.ranks()
    assert expectations["SLC16A1"] is RankExpectation.DISCREPANT
    assert panel.labels()["SLC16A1"] == 1
    assert CALIBRATION_GENES == ("SLC2A1", "SLC16A9", "CPT1A", "CES1")
    assert (
        "crossed the pharmacological threshold only"
        in next(gene for gene in CONTROL_GENES if gene.symbol == "SLC16A1").rationale
    )


def test_control_panel_observations_are_complete() -> None:
    panel = build_control_panel()
    for modality in MODALITY_ORDER:
        observations = modality_observations(panel, modality)
        assert len(observations) == len(panel.symbols())
        assert all(0.0 <= value <= 1.0 for _, value in observations)


def test_control_panel_is_deterministic_for_a_fixed_seed() -> None:
    first = build_control_panel(seed=11)
    second = build_control_panel(seed=11)
    assert first.observations == second.observations
    third = build_control_panel(seed=12)
    assert first.observations != third.observations


def test_calibration_builds_four_leave_one_out_folds(control_panel) -> None:
    folds = build_folds(control_panel)
    assert len(folds) == 4
    audit = audit_folds(control_panel, folds)
    assert audit["every_symbol_withheld_once"]
    assert audit["withheld_positives"] == ["SLC2A1"]
    assert audit["never_calibrated"] == ["SLC16A1"]
    assert fold_for_symbol(folds, "CPT1A").held_out_symbol == "CPT1A"
    assert len(withheld_positives(control_panel, folds)) == 1


def test_full_panel_policy_yields_a_single_fold(control_panel) -> None:
    folds = build_folds(control_panel, FoldPolicy.FULL_PANEL)
    assert len(folds) == 1
    assert folds[0].held_out_symbol is None


def test_closed_form_rate_minimises_the_panel_loss() -> None:
    """The closed form must beat a brute-force grid on the same panel."""

    grid = np.linspace(0.0, 1.0, 401)
    separable_scores = np.linspace(0.05, 0.95, 11)
    separable_labels = (separable_scores > 0.5).astype(float)
    noisy_scores = np.asarray([0.92, 0.81, 0.68, 0.31, 0.44, 0.18, 0.77, 0.22])
    noisy_labels = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    for scores, labels in ((separable_scores, separable_labels), (noisy_scores, noisy_labels)):
        closed = fit_rate_closed_form(scores, labels)
        grid_losses = np.asarray([rate_loss(scores, labels, value) for value in grid])
        assert rate_loss(scores, labels, closed) <= float(np.min(grid_losses)) + 1e-9
        numeric = fit_rate_numeric(scores, labels)
        assert rate_loss(scores, labels, numeric) <= float(np.min(grid_losses)) + 1e-6
        assert 0.0 <= closed <= 1.0
        assert fit_rate_closed_form(noisy_scores, noisy_labels) <= 1.0


def test_closed_form_and_numeric_minimisers_agree(control_panel) -> None:
    for modality in MODALITY_ORDER:
        observations = modality_observations(control_panel, modality)
        labels = control_panel.labels()
        scores = np.asarray([value for _, value in observations], dtype=float)
        targets = np.asarray([float(labels[symbol]) for symbol, _ in observations], dtype=float)
        assert closed_form_agreement(scores, targets) < 1e-3
        assert 0.0 <= fit_rate_numeric(scores, targets) <= 1.0


def test_single_modality_pignistic_hand_computed() -> None:
    assert single_modality_pignistic(1.0, 1.0) == pytest.approx(1.0)
    assert single_modality_pignistic(0.0, 1.0) == pytest.approx(0.0)
    assert single_modality_pignistic(0.5, 0.0) == pytest.approx(0.5)
    assert single_modality_pignistic(0.0, 0.5) == pytest.approx(0.25)


def test_calibration_report_is_complete(control_panel) -> None:
    report = calibrate(control_panel)
    payload = report.as_dict()
    assert len(payload["folds"]) == 4
    assert set(payload["pooled_rates"]) == {modality.value for modality in MODALITY_ORDER}
    assert 0.0 <= payload["ignorance_floor"] <= 1.0
    assert payload["reliability_order"]
    assert len(payload["held_out_rate_provenance"]) >= 1


def test_fitted_rates_do_not_worsen_the_panel(control_panel) -> None:
    report = calibrate(control_panel)
    for fold in report.folds:
        for fit_result in fold.fits:
            assert fit_result.loss <= fit_result.brier_at_unit_rate + 1e-9


def test_mass_map_forward_pass_is_a_valid_triple() -> None:
    import torch

    module = CalibratedMassMap()
    scores = torch.tensor([[0.0, 0.5, 0.9, 1.0]], dtype=torch.float32)
    masses = module(scores)
    assert masses.shape == (1, 4, 3)
    totals = masses.sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)
    assert torch.all(masses >= 0.0)


def test_mass_map_parameters_convert_to_a_plain_mapper() -> None:
    from crb_concordance.agents.mapping import MassMapper

    module = CalibratedMassMap()
    mapper = MassMapper(module.to_params())
    triple = mapper.map_score(Modality.KG, 0.8)
    assert 0.0 <= triple.pignistic() <= 1.0
    assert sum(triple.as_tuple()) == pytest.approx(1.0)


def test_brier_mass_loss_is_zero_for_a_perfect_fit() -> None:
    import torch

    module = CalibratedMassMap()
    with torch.no_grad():
        module.slope.fill_(60.0)
        module.offset.fill_(-30.0)
    scores = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    labels = torch.ones((1, 1), dtype=torch.float32)
    mask = torch.ones((1, 4), dtype=torch.bool)
    loss = brier_mass_loss(module, scores, labels, mask)
    assert float(loss) < 3e-3


def test_calibration_table_masks_absent_observations(control_panel) -> None:
    table = build_table(control_panel)
    assert table.observations == len(control_panel.symbols())
    assert table.mask.shape == (table.observations, len(MODALITY_ORDER))
    assert table.mask.all()
    batches = batches_from_table(table, 2)
    assert sum(batch[0].shape[0] for batch in batches) == table.observations


def test_fitting_the_mass_map_improves_the_panel(tmp_path) -> None:
    panel = build_control_panel()
    config = OptimConfig(learning_rate=0.05, epochs=150, batch_size=5, seed=3)
    report = fit_mass_map(panel, config=config, checkpoint_path=tmp_path / "map.ckpt")
    assert report.after["brier"] <= report.before["brier"] + 1e-12
    assert report.history.decreased()
    assert report.mapper.as_dict()


def test_engine_reduces_the_loss_on_a_fixed_batch(tmp_path) -> None:
    panel = build_control_panel()
    table = build_table(panel)
    batches = batches_from_table(table, table.observations)
    module = CalibratedMassMap()
    initial = evaluate_loss(module, batches, brier_mass_loss)
    history = fit(
        module,
        batches,
        brier_mass_loss,
        OptimConfig(learning_rate=0.05, epochs=120, batch_size=5, warmup_fraction=0.0, seed=2),
        checkpoint_path=tmp_path / "model.ckpt",
        ema=False,
    )
    assert history.final_loss < initial
    assert history.records[-1].step == len(history.records) - 1


def test_checkpoint_restores_the_seed_and_payload(tmp_path) -> None:
    module = CalibratedMassMap()
    parameters = {
        name: value.detach().cpu().numpy().copy() for name, value in module.named_parameters()
    }
    path = tmp_path / "state.ckpt"
    save_meta = fit(
        module,
        batches_from_table(build_table(build_control_panel()), 5),
        brier_mass_loss,
        OptimConfig(epochs=3, batch_size=5, seed=17),
        checkpoint_path=path,
        ema=False,
    )
    assert save_meta.checkpoint is not None
    restored, meta = load_checkpoint(path)
    assert meta.seed == 17
    assert set(restored) == set(parameters)
    assert resume_seed(path) == 17


def test_checkpoint_digest_detects_tampering(tmp_path) -> None:
    path = tmp_path / "state.ckpt"
    fit(
        CalibratedMassMap(),
        batches_from_table(build_table(build_control_panel()), 5),
        brier_mass_loss,
        OptimConfig(epochs=2, batch_size=5, seed=1),
        checkpoint_path=path,
        ema=False,
    )
    raw = path.read_bytes()
    marker = raw.find(b'"digest"')
    assert marker > 0
    head = raw[: marker + len('"digest": "')]
    tail = raw[marker + len('"digest": "') + 1 :]
    path.write_bytes(head + b"0" + tail)
    with pytest.raises(CheckpointError):
        load_checkpoint(path)
    path.write_bytes(raw)


def test_schedule_warmup_then_decay() -> None:
    config = OptimConfig(epochs=4, warmup_fraction=0.25)
    steps = total_steps(config, samples=40)
    assert steps == 4 * 2
    curve = schedule_curve(config, steps_per_epoch=2)
    assert curve.size == steps
    assert curve[0] < curve[2]
    assert curve[-1] <= curve[0]
    assert learning_rate_at(0, 100, config) == pytest.approx(config.learning_rate * 0.04)
    assert learning_rate_at(100, 100, config) == pytest.approx(config.min_learning_rate)


def test_evidence_design_rejects_inconsistent_parameters() -> None:
    with pytest.raises(ValueError):
        EvidenceDesign(prevalence=1.5).validate()
    with pytest.raises(ValueError):
        EvidenceDesign(conflict_probability=1.5).validate()
    with pytest.raises(ValueError):
        EvidenceDesign(absent_fraction=-0.1).validate()


def test_simulated_candidates_hold_the_declared_prevalence() -> None:
    design = EvidenceDesign(n_candidates=1000, prevalence=0.2, conflict_probability=0.3, seed=9)
    candidates = simulate_candidates(design)
    positives = sum(1 for candidate in candidates if candidate.vulnerable)
    assert positives == 200
    assert all(set(candidate.scores) == set(MODALITY_ORDER) for candidate in candidates)


def test_absent_fraction_removes_evidence() -> None:
    design = EvidenceDesign(n_candidates=400, prevalence=0.15, absent_fraction=0.5, seed=4)
    candidates = simulate_candidates(design)
    absent = sum(
        1
        for candidate in candidates
        for modality in MODALITY_ORDER
        if candidate.scores[modality] is None
    )
    total = len(candidates) * len(MODALITY_ORDER)
    assert 0.4 * total <= absent <= 0.6 * total
