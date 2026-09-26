"""Fit the discount rates and the evidence-to-mass map.

Ref: Sec. 4.1 (the rates are fitted on the retrospective control panel through the
four-fold panel); Sec. 4.2, Algorithm 3 (the fitted mapping and rates are then fixed
for discovery time); Supplementary Table S1 (the panel is a design parameter).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crb_concordance.calibration.discount_fit import (
    calibrate,
    closed_form_agreement,
    rate_stability,
)
from crb_concordance.calibration.folds import audit_folds, build_folds
from crb_concordance.calibration.mass_calibration import fit_mass_map
from crb_concordance.cli.arguments import common_parser, experiment_output
from crb_concordance.cohorts.control_panel import (
    ControlPanel,
    modality_observations,
)
from crb_concordance.evaluation.reporting import render_text
from crb_concordance.training.optim import OptimConfig
from crb_concordance.utils.atomic import atomic_write_json, atomic_write_text
from crb_concordance.utils.logging import configure_logging, get_logger
from crb_concordance.utils.types import MODALITY_ORDER, Modality

LOGGER = get_logger("cli.calibrate")


def _agreement(panel: ControlPanel, modality: Modality) -> float:
    """Gap between the closed-form and the derivative-free rate on the whole panel."""

    observations = modality_observations(panel, modality)
    labels = panel.labels()
    scores = np.asarray([value for _, value in observations], dtype=float)
    targets = np.asarray([float(labels[symbol]) for symbol, _ in observations], dtype=float)
    return closed_form_agreement(scores, targets)


def run(
    *,
    config_root: str | Path,
    experiment: str,
    overrides: list[str],
    output_root: str | Path,
) -> dict[str, object]:
    from crb_concordance.cli.runtime import build_runtime

    runtime = build_runtime(config_root=config_root, experiment=experiment, overrides=overrides)
    panel = runtime.panel
    folds = build_folds(panel)
    report = calibrate(panel)
    optim = OptimConfig(
        learning_rate=float(runtime.config.get("train.learning_rate", 0.05)),
        epochs=int(runtime.config.get("train.epochs", 400)),
        batch_size=int(runtime.config.get("train.batch_size", 5)),
        seed=int(runtime.config.get("train.seed", 20260101)),
    )
    directory = experiment_output(output_root, experiment)
    mass_report = fit_mass_map(panel, config=optim, checkpoint_path=directory / "mass_map.ckpt")
    table = mass_report.mapper
    payload: dict[str, object] = {
        "panel": {"symbols": list(panel.symbols()), "scale_note": panel.scale_note},
        "folds": audit_folds(panel, folds),
        "discount_rates": {
            "pooled": report.pooled.as_dict(),
            "floor": report.pooled.ignorance_floor(),
        },
        "fold_rates": [fold.as_dict() for fold in report.folds],
        "rate_stability": rate_stability(report),
        "closed_form_agreement": [
            {"modality": modality.value, "agreement": _agreement(panel, modality)}
            for modality in MODALITY_ORDER
        ],
        "mass_map": {
            "before": mass_report.before,
            "after": mass_report.after,
            "parameters": table.as_dict(),
            "steps": len(mass_report.history.records),
            "loss_decreased": mass_report.history.decreased(),
        },
    }
    atomic_write_json(directory / "calibration_report.json", payload)
    atomic_write_text(
        directory / "calibration_report.txt",
        render_text("Calibration report", payload),
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("fit discount rates and the evidence-to-mass map")
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else "INFO")
    payload = run(
        config_root=args.config_root,
        experiment=args.experiment,
        overrides=args.override,
        output_root=args.output_root,
    )
    LOGGER.info("pooled discount rates %s", payload["discount_rates"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
