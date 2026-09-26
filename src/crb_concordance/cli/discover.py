"""Run the end-to-end discovery pass and write the ranking and the ledger.

Ref: Sec. 4.2, Algorithm 1 (for each candidate collect the four evidences, discount,
combine, append the per-agent entries and the belief quantities, and return the pool
ranked by BetP(V) descending with intervals and traces attached).
"""

from __future__ import annotations

from pathlib import Path

from crb_concordance.cli.arguments import common_parser, experiment_output
from crb_concordance.cli.runtime import build_runtime
from crb_concordance.cohorts.control_panel import ControlPanel
from crb_concordance.discovery.pipeline import DiscoveryConfig, run_discovery
from crb_concordance.discovery.ranking import separability
from crb_concordance.evaluation.reporting import render_text
from crb_concordance.utils.atomic import atomic_write_json, atomic_write_text
from crb_concordance.utils.logging import configure_logging, get_logger

LOGGER = get_logger("cli.discover")

CONTROL_SEPARABILITY_CEILING = 0.3


def run(
    *,
    config_root: str | Path,
    experiment: str,
    overrides: list[str],
    output_root: str | Path,
    pool_limit: int | None = None,
) -> dict[str, object]:
    runtime = build_runtime(config_root=config_root, experiment=experiment, overrides=overrides)
    pool = runtime.candidate_pool()
    if pool_limit is not None:
        pool = pool[:pool_limit]
    config = DiscoveryConfig(
        conflict_threshold=float(runtime.config.get("model.conflict_threshold", 0.05)),
        max_hops=int(runtime.config.get("model.max_hops", 3)),
    )
    result = run_discovery(
        pool, runtime.context, runtime.rates, config=config, mapper=runtime.mapper
    )
    directory = experiment_output(output_root, experiment)
    ledger_path = result.ledger.flush(directory / "provenance_ledger.jsonl")
    panel: ControlPanel = runtime.panel
    ranks = {entry.candidate: entry.rank for entry in result.ranked}
    control_ranks = {symbol: ranks.get(symbol) for symbol in panel.symbols()}
    payload: dict[str, object] = {
        "rates": runtime.rates.as_dict(),
        "config": config.as_dict(),
        "structural": result.structural,
        "provenance": result.provenance(),
        "control_ranks": control_ranks,
        "control_panel": {
            "positives": list(panel.positives()),
            "negatives": list(panel.negatives()),
        },
        "separability": separability(result.ranked, width_ceiling=CONTROL_SEPARABILITY_CEILING),
        "top": [entry.as_row() for entry in result.ranked[:20]],
        "ledger_rows": len(result.ledger),
        "ledger_path": str(ledger_path),
    }
    atomic_write_json(directory / "discovery_report.json", payload)
    atomic_write_text(directory / "discovery_report.txt", render_text("Discovery report", payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("run the discovery pass over the substrate candidate pool")
    parser.add_argument("--pool-limit", type=int, default=None)
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else "INFO")
    payload = run(
        config_root=args.config_root,
        experiment=args.experiment,
        overrides=args.override,
        output_root=args.output_root,
        pool_limit=args.pool_limit,
    )
    LOGGER.info("ranked %d candidates", len(payload["top"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
