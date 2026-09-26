"""Execute the benchmark harness on this release's own substrate pool.

Ref: Table 1 (every outcome cell is pending until the cohort arms are analysed, and
each row marked for reimplementation is scored on this paper's own temporal-holdout
candidate pool with identical preprocessing); Table 2 (the clinical-signature
comparison is pending for the same reason); Sec. 4.4 (five metrics on the held-out
pool, with the hit rate of a named control reported individually).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from crb_concordance.agents.base import EvidenceAgent
from crb_concordance.agents.dependency import DependencyAgent
from crb_concordance.agents.flux_feasibility import FluxFeasibilityAgent
from crb_concordance.agents.kg_path import KGAgentConfig, KnowledgeGraphAgent
from crb_concordance.agents.transcriptomic_clinical import ClinicalAssociationAgent
from crb_concordance.belief.discount import discount
from crb_concordance.benchmark.baselines.operators import (
    OPERATORS,
    score_pool,
    validate_bindings,
)
from crb_concordance.benchmark.pool import assemble_pool, pool_size_summary
from crb_concordance.benchmark.registry import FUSION_PROPOSED, validate_registry
from crb_concordance.cli.arguments import common_parser, experiment_output
from crb_concordance.cli.runtime import StudyRuntime, build_runtime
from crb_concordance.discovery.pipeline import DiscoveryConfig, DiscoveryResult, run_discovery
from crb_concordance.evaluation.reporting import (
    render_text,
    table_one_snapshot,
    table_two_snapshot,
)
from crb_concordance.metrics.ranking import (
    evaluate_ranking,
    hit_rate,
    rank_of,
    rank_order,
)
from crb_concordance.simulation.evidence_model import CandidateSummary
from crb_concordance.utils.atomic import atomic_write_json, atomic_write_text
from crb_concordance.utils.logging import configure_logging, get_logger
from crb_concordance.utils.types import MODALITY_ORDER, Modality

LOGGER = get_logger("cli.benchmark")

DEFAULT_FOLDS = 5


def run(
    *,
    config_root: str | Path,
    experiment: str,
    overrides: list[str],
    output_root: str | Path,
    seed: int = 1,
) -> dict[str, object]:
    runtime = build_runtime(config_root=config_root, experiment=experiment, overrides=overrides)
    pool = assemble_pool(runtime.substrate, runtime.panel, runtime.finder)
    positives = pool.positive_set()
    candidate_pool = tuple(set(runtime.candidate_pool()) | set(runtime.panel.symbols()))
    summaries = _summaries(runtime, candidate_pool)
    proposed_scores = {
        entry.candidate: entry.pignistic for entry in _discovery(runtime, candidate_pool).ranked
    }
    executed: dict[str, dict[str, float]] = {}
    ranking_rows: list[dict[str, object]] = []
    for operator in OPERATORS:
        scores = score_pool(summaries, operator, seed=seed)
        metrics = evaluate_ranking(scores, positives, k=10)
        executed[operator.row] = {
            "Recall@10": metrics.recall_at_k,
            "Prec.@10": metrics.precision_at_k,
            "MRR": metrics.mean_reciprocal_rank,
            "Prov. (%)": runtime.rates.ignorance_floor(),
        }
        ranking_rows.append(
            {
                "method": operator.row,
                "family": operator.family.value,
                "aggregation": operator.aggregation.value,
                "modalities": [modality.value for modality in operator.modalities],
                "recall_at_10": metrics.recall_at_k,
                "precision_at_10": metrics.precision_at_k,
                "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
                "average_precision": metrics.average_precision,
            }
        )
    proposed_metrics = evaluate_ranking(proposed_scores, positives, k=10)
    control_hits = {
        symbol: hit_rate(rank_order(proposed_scores), symbol, 10)
        for symbol in runtime.panel.symbols()
    }
    control_ranks = {symbol: rank_of(proposed_scores, symbol) for symbol in runtime.panel.symbols()}
    payload: dict[str, object] = {
        "registry": validate_registry(),
        "bindings": validate_bindings(),
        "pool": pool.as_dict(),
        "pool_size": pool_size_summary(runtime.substrate),
        "modalities": [modality.value for modality in MODALITY_ORDER],
        "substrate_pool_ranking": ranking_rows,
        "proposed": {
            "method": FUSION_PROPOSED,
            "recall_at_10": proposed_metrics.recall_at_k,
            "precision_at_10": proposed_metrics.precision_at_k,
            "mean_reciprocal_rank": proposed_metrics.mean_reciprocal_rank,
            "average_precision": proposed_metrics.average_precision,
        },
        "control_hits_at_10": control_hits,
        "control_ranks": control_ranks,
        "table_one": table_one_snapshot(),
        "table_two": table_two_snapshot(),
        "labelling_note": (
            "the substrate_pool_ranking values are produced on this release's own synthetic "
            "substrate stand-in; Table 1 and Table 2 cells stay pending because their values "
            "require the cohort arms, which are not redistributed"
        ),
    }
    directory = experiment_output(output_root, experiment)
    atomic_write_json(directory / "benchmark_report.json", payload)
    atomic_write_text(directory / "benchmark_report.txt", render_text("Benchmark report", payload))
    return payload


def _summaries(
    runtime: StudyRuntime, candidate_pool: tuple[str, ...]
) -> tuple[CandidateSummary, ...]:
    """Per-candidate belief summaries for the operator harness, from the real agents."""

    context = runtime.context
    rates = runtime.rates
    mapper = runtime.mapper
    agents: dict[Modality, EvidenceAgent] = {
        Modality.KG: KnowledgeGraphAgent(
            context.finder,
            context.phenotype,
            config=KGAgentConfig(max_hops=context.finder.max_hops),
        ),
        Modality.DEP: DependencyAgent(context.dependency_table),
        Modality.FLUX: FluxFeasibilityAgent(context.model),
        Modality.CLIN: ClinicalAssociationAgent(context.profiles, context.labels_by_record()),
    }
    summaries: list[CandidateSummary] = []
    for candidate in candidate_pool:
        evidences = {modality: agent.collect(candidate) for modality, agent in agents.items()}
        discounted = {
            modality: discount(mapper.map_evidence(evidences[modality]), rates.of(modality))
            for modality in MODALITY_ORDER
        }
        pignistics = np.asarray(
            [discounted[modality].pignistic() for modality in MODALITY_ORDER], dtype=float
        )
        widths = np.asarray(
            [discounted[modality].interval_width() for modality in MODALITY_ORDER], dtype=float
        )
        summaries.append(
            CandidateSummary(
                gene=candidate,
                vulnerable=candidate in runtime.panel.positives(),
                pignistic=float(np.mean(pignistics)) if pignistics.size else 0.5,
                interval_width=float(np.mean(widths)) if widths.size else 1.0,
                total_conflict=0.0,
                modality_pignistic={
                    modality: discounted[modality].pignistic() for modality in MODALITY_ORDER
                },
                absent_modalities=sum(
                    1 for modality in MODALITY_ORDER if not evidences[modality].present
                ),
                flagged_steps=0,
                trace_steps=0,
            )
        )
    return tuple(summaries)


def _discovery(runtime: StudyRuntime, candidate_pool: tuple[str, ...]) -> DiscoveryResult:
    return run_discovery(
        candidate_pool,
        runtime.context,
        runtime.rates,
        config=DiscoveryConfig(
            conflict_threshold=float(runtime.config.get("model.conflict_threshold", 0.05)),
            max_hops=int(runtime.config.get("model.max_hops", 3)),
        ),
        mapper=runtime.mapper,
    )


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("execute the benchmark harness on the substrate pool")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else "INFO")
    payload = run(
        config_root=args.config_root,
        experiment=args.experiment,
        overrides=args.override,
        output_root=args.output_root,
        seed=args.seed,
    )
    LOGGER.info("baseline rows %d", len(payload["substrate_pool_ranking"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
