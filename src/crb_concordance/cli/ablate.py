"""Run the declared ablation battery and the substitution tier.

Ref: Sec. 4.5 (four levels of extraction plus a substitution tier; the report also
records what happens to the conflict-trace signal when it is removed).
"""

from __future__ import annotations

from pathlib import Path

from crb_concordance.calibration.discount_fit import calibrate
from crb_concordance.cli.arguments import common_parser, experiment_output
from crb_concordance.cohorts.control_panel import build_control_panel
from crb_concordance.cohorts.generator import EvidenceDesign, simulate_candidates
from crb_concordance.evaluation.ablations import full_plan, run_battery
from crb_concordance.evaluation.reporting import render_text
from crb_concordance.utils.atomic import atomic_write_json, atomic_write_text
from crb_concordance.utils.logging import configure_logging, get_logger
from crb_concordance.utils.types import MODALITY_ORDER, Modality

LOGGER = get_logger("cli.ablate")

DEFAULT_POOL = 600
DEFAULT_CUTOFF = 30


def run(
    *,
    experiment: str,
    output_root: str | Path,
    n_candidates: int = DEFAULT_POOL,
    conflict_probability: float = 0.3,
    absent_fraction: float = 0.0,
    seed: int = 1,
    cutoff: int = DEFAULT_CUTOFF,
) -> dict[str, object]:
    candidates = simulate_candidates(
        EvidenceDesign(
            n_candidates=n_candidates,
            prevalence=0.15,
            conflict_probability=conflict_probability,
            absent_fraction=absent_fraction,
            seed=seed,
        )
    )
    rates = calibrate(build_control_panel()).pooled
    outcomes = run_battery(candidates, rates, cutoff=cutoff)
    payload: dict[str, object] = {
        "plan": full_plan(),
        "candidate_pool": n_candidates,
        "conflict_probability": conflict_probability,
        "absent_fraction": absent_fraction,
        "cutoff": cutoff,
        "rates": rates.as_dict(),
        "outcomes": [outcome.as_dict() for outcome in outcomes],
        "modalities": [modality.value for modality in MODALITY_ORDER],
        "single_agent_default": Modality.DEP.value,
    }
    directory = experiment_output(output_root, experiment)
    atomic_write_json(directory / "ablation_report.json", payload)
    atomic_write_text(directory / "ablation_report.txt", render_text("Ablation report", payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("run the ablation battery over a simulated pool")
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_POOL)
    parser.add_argument("--conflict-probability", type=float, default=0.3)
    parser.add_argument("--absent-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else "INFO")
    payload = run(
        experiment=args.experiment,
        output_root=args.output_root,
        n_candidates=args.n_candidates,
        conflict_probability=args.conflict_probability,
        absent_fraction=args.absent_fraction,
        seed=args.seed,
        cutoff=args.cutoff,
    )
    LOGGER.info("ablation cells %d", len(payload["outcomes"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
