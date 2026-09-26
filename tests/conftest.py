"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = RELEASE_ROOT / "configs"


@pytest.fixture(scope="session")
def release_root() -> Path:
    return RELEASE_ROOT


@pytest.fixture(scope="session")
def config_root() -> Path:
    return CONFIG_ROOT


@pytest.fixture(scope="session")
def learned_rates():
    from crb_concordance.calibration.discount_fit import calibrate
    from crb_concordance.cohorts.control_panel import build_control_panel

    return calibrate(build_control_panel()).pooled


@pytest.fixture(scope="session")
def control_panel():
    from crb_concordance.cohorts.control_panel import build_control_panel

    return build_control_panel()


@pytest.fixture(scope="session")
def substrate():
    from crb_concordance.graph.generator import build_substrate

    return build_substrate()


@pytest.fixture(scope="session")
def flux_model():
    from crb_concordance.metabolism.model import StoichiometricModel

    return StoichiometricModel.from_catalogue()


@pytest.fixture(scope="session")
def cohort_records():
    from crb_concordance.cohorts.generator import CohortDesign, generate_cohort

    return generate_cohort(CohortDesign())


@pytest.fixture(scope="session")
def small_pool():
    from crb_concordance.cohorts.generator import EvidenceDesign, simulate_candidates

    return simulate_candidates(
        EvidenceDesign(
            n_candidates=300,
            prevalence=0.15,
            conflict_probability=0.3,
            absent_fraction=0.25,
            seed=5,
        )
    )
