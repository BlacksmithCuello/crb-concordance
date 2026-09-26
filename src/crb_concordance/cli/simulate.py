"""Run the in silico simulation study and emit its contrasts.

Ref: Sec. 2 (the in silico simulation is the study carried out, and its results are
what the program in the Methods produced); Sec. 4.4 (the pre-specified falsification
inequality); Sec. 4.5 (H1 and H2); Supplementary Table S1 (the grids and seeds).
"""

from __future__ import annotations

from pathlib import Path

from crb_concordance.cli.arguments import common_parser, experiment_output
from crb_concordance.cli.runtime import build_runtime
from crb_concordance.evaluation.hypotheses import build_hypothesis_report
from crb_concordance.evaluation.reporting import render_text
from crb_concordance.simulation.contrasts import (
    all_cell_contrasts,
    pooled_trace_analysis,
)
from crb_concordance.simulation.design import (
    ABSENT_GRID,
    CONFLICT_GRID,
    PARITY_MARGIN,
    PRIMARY_RECALL_CUTOFF,
    SEED_BOOTSTRAP_RESAMPLES,
    SIMULATED_CANDIDATES,
    SIMULATED_PREVALENCE,
    SIMULATION_SEEDS,
    SimulationCell,
    SimulationDesign,
)
from crb_concordance.simulation.runner import run_design
from crb_concordance.stats.power import prospective_sizing
from crb_concordance.utils.atomic import atomic_write_json, atomic_write_text
from crb_concordance.utils.config import ExperimentConfig
from crb_concordance.utils.logging import configure_logging, get_logger

LOGGER = get_logger("cli.simulate")


def build_design(
    *,
    seeds: int | None,
    candidates: int | None,
    bootstrap: int | None,
    config: ExperimentConfig | None = None,
) -> SimulationDesign:
    base = SimulationDesign()
    if config is not None:
        base = _design_from_config(config)
    if seeds is None and candidates is None and bootstrap is None:
        return base
    cells = tuple(
        SimulationCell(
            conflict_probability=cell.conflict_probability,
            absent_fraction=cell.absent_fraction,
            seeds=cell.seeds if seeds is None else cell.seeds[:seeds],
            n_candidates=cell.n_candidates if candidates is None else candidates,
        )
        for cell in base.cells
    )
    return SimulationDesign(
        cells=cells,
        bootstrap_resamples=base.bootstrap_resamples if bootstrap is None else bootstrap,
    )


def _design_from_config(config: ExperimentConfig) -> SimulationDesign:
    """Rebuild the grid from the experiment block, keeping the seeds unless overridden."""

    conflicts = config.get("simulation.conflict_probability_grid", CONFLICT_GRID)
    absent = config.get("simulation.absent_evidence_fraction_grid", ABSENT_GRID)
    candidates = int(config.get("simulation.candidates_per_seed", SIMULATED_CANDIDATES))
    prevalence = float(config.get("simulation.prevalence", SIMULATED_PREVALENCE))
    seed_count = int(config.get("simulation.seeds", len(SIMULATION_SEEDS)))
    cells = tuple(
        SimulationCell(
            conflict_probability=float(conflict),
            absent_fraction=float(fraction),
            seeds=tuple(range(1, seed_count + 1)),
            n_candidates=candidates,
            prevalence=prevalence,
        )
        for conflict in conflicts
        for fraction in absent
    )
    return SimulationDesign(
        cells=cells,
        bootstrap_resamples=int(
            config.get("simulation.seed_level_bootstrap_resamples", SEED_BOOTSTRAP_RESAMPLES)
        ),
        recall_cutoff=int(config.get("simulation.recall_cutoff", PRIMARY_RECALL_CUTOFF)),
        parity_margin=float(config.get("simulation.parity_margin", PARITY_MARGIN)),
    )


def run(
    *,
    config_root: str | Path,
    experiment: str,
    overrides: list[str],
    output_root: str | Path,
    seeds: int | None = None,
    candidates: int | None = None,
    bootstrap: int | None = None,
    pooled_seeds: int = 2,
) -> dict[str, object]:
    runtime = build_runtime(config_root=config_root, experiment=experiment, overrides=overrides)
    design = build_design(
        seeds=seeds, candidates=candidates, bootstrap=bootstrap, config=runtime.config
    )
    design.validate()
    result = run_design(runtime.rates, design=design, mapper=runtime.mapper)
    hypotheses = build_hypothesis_report(runtime.rates, result, prospective_sizing())
    contrasts = all_cell_contrasts(result)
    pooled = pooled_trace_analysis(
        runtime.rates,
        design,
        mapper=runtime.mapper,
        seeds_per_cell=pooled_seeds,
        flag_threshold=float(runtime.config.get("model.conflict_threshold", 0.05)),
    )
    payload: dict[str, object] = {
        "design": design.as_dict(),
        "rates": runtime.rates.as_dict(),
        "cells": [contrast.as_dict() for contrast in contrasts],
        "runs": [run.as_dict() for run in result.runs()],
        "pooled_trace_analysis": pooled,
        "hypotheses": hypotheses.as_dict(),
        "provenance_note": (
            "the simulation is executed here; the abstract's separability and conflict-mass "
            "figures are reported without a stated estimator, so they are not reproduced"
        ),
    }
    directory = experiment_output(output_root, experiment)
    atomic_write_json(directory / "simulation_report.json", payload)
    atomic_write_text(
        directory / "simulation_report.txt", render_text("Simulation report", payload)
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("run the in silico simulation study")
    parser.add_argument("--seeds", type=int, default=None, help="seeds per grid cell")
    parser.add_argument("--candidates", type=int, default=None, help="candidates per seed")
    parser.add_argument("--bootstrap", type=int, default=None, help="bootstrap resamples")
    parser.add_argument("--pooled-seeds", type=int, default=2)
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else "INFO")
    payload = run(
        config_root=args.config_root,
        experiment=args.experiment,
        overrides=args.override,
        output_root=args.output_root,
        seeds=args.seeds,
        candidates=args.candidates,
        bootstrap=args.bootstrap,
        pooled_seeds=args.pooled_seeds,
    )
    LOGGER.info("simulation runs %s", payload["design"]["total_runs"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
