"""Compute the pre-specified prospective-arm sizing table.

Ref: Sec. 4.6 (a Hanley-McNeil-based power calculation for two correlated AUROC
curves measured on the same cohort, with the sensitivity of the result to the
assumed correlation reported explicitly).
"""

from __future__ import annotations

from pathlib import Path

from crb_concordance.cli.arguments import add_sizing_arguments, common_parser, experiment_output
from crb_concordance.evaluation.reporting import render_text
from crb_concordance.stats.power import SizingInputs, accrual_range_covers, prospective_sizing
from crb_concordance.utils.atomic import atomic_write_json, atomic_write_text
from crb_concordance.utils.logging import configure_logging, get_logger

LOGGER = get_logger("cli.size_prospective")


def run(
    *,
    experiment: str,
    output_root: str | Path,
    baseline_auroc: float,
    margin: float,
    prevalence: float,
    alpha: float,
    power: float,
) -> dict[str, object]:
    inputs = SizingInputs(
        baseline_auroc=baseline_auroc,
        margin=margin,
        prevalence=prevalence,
        alpha=alpha,
        power=power,
    )
    result = prospective_sizing(inputs)
    payload: dict[str, object] = {
        "inputs": inputs.as_dict(),
        "rows": result.as_rows(),
        "accrual_coverage": accrual_range_covers(result),
    }
    directory = experiment_output(output_root, experiment)
    atomic_write_json(directory / "sizing_report.json", payload)
    atomic_write_text(directory / "sizing_report.txt", render_text("Prospective sizing", payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("compute the prospective-arm sizing table")
    add_sizing_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else "INFO")
    payload = run(
        experiment=args.experiment,
        output_root=args.output_root,
        baseline_auroc=args.baseline_auroc,
        margin=args.margin,
        prevalence=args.prevalence,
        alpha=args.alpha,
        power=args.power,
    )
    LOGGER.info("sizing rows %d", len(payload["rows"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
