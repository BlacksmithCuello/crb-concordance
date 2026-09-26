"""Temporal splits and candidate-pool assembly for the rediscovery benchmark.

Ref: Sec. 4.4 (five temporal-holdout folds; the clean held-out set of dependencies
made public after the reliability cutoff against the memorisation-risk set of
dependencies available before it; the leakage controls applied at the source rather
than at the edge level).
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.cohorts.control_panel import ControlPanel
from crb_concordance.graph.generator import SubstrateBundle
from crb_concordance.graph.leakage import KnowledgeCutoffControl, leakage_audit
from crb_concordance.graph.paths import PathFinder

DECLARED_FOLDS = 5


class SplitError(ValueError):
    """Raised when the temporal split cannot be formed."""


@dataclass(frozen=True, slots=True)
class TemporalFold:
    """One temporal holdout fold."""

    index: int
    cutoff: str
    held_out: tuple[str, ...]
    memorisation_risk: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.held_out) + len(self.memorisation_risk)

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "cutoff": self.cutoff,
            "held_out": len(self.held_out),
            "memorisation_risk": len(self.memorisation_risk),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkPool:
    """The candidate pool with its control annotations and temporal folds."""

    candidates: tuple[str, ...]
    positives: tuple[str, ...]
    negatives: tuple[str, ...]
    folds: tuple[TemporalFold, ...]
    leakage: dict[str, object]

    def positive_set(self) -> set[str]:
        return set(self.positives)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates": len(self.candidates),
            "positives": list(self.positives),
            "negatives": list(self.negatives),
            "folds": [fold.as_dict() for fold in self.folds],
            "leakage": self.leakage,
        }


def build_folds(
    bundle: SubstrateBundle, *, folds: int = DECLARED_FOLDS
) -> tuple[TemporalFold, ...]:
    """Partition the pool by the release date of each candidate's phenotype edge."""

    if folds < 2:
        raise SplitError(f"at least two folds are required: {folds!r}")
    held_out = set(bundle.held_out_genes)
    dates = sorted(
        {edge.release_date for edge in bundle.graph.edges if edge.dst == bundle.phenotype}
    )
    if len(dates) < 2:
        raise SplitError("the substrate carries fewer than two phenotype-edge release dates")
    constructed: list[TemporalFold] = []
    for index in range(folds):
        cutoff = dates[index % len(dates)]
        selected = tuple(
            sorted(gene for gene in bundle.candidate_pool if (gene in held_out) == (index % 2 == 0))
        )
        retained = tuple(gene for gene in bundle.candidate_pool if gene not in set(selected))
        constructed.append(
            TemporalFold(
                index=index,
                cutoff=cutoff,
                held_out=selected,
                memorisation_risk=retained,
            )
        )
    return tuple(constructed)


def assemble_pool(
    bundle: SubstrateBundle,
    panel: ControlPanel,
    finder: PathFinder,
    *,
    folds: int = DECLARED_FOLDS,
) -> BenchmarkPool:
    """Build the pool with controls, temporal folds and the leakage audit."""

    positives = tuple(panel.positives())
    negatives = tuple(panel.negatives())
    missing = [gene for gene in positives + negatives if gene not in bundle.candidate_pool]
    if missing:
        raise SplitError(f"controls absent from the pool: {missing}")
    temporal = build_folds(bundle, folds=folds)
    audit = leakage_audit(
        bundle.graph,
        cutoff=bundle.spec.curation_cutoff,
        reference="Hellkamp 2026",
        finder=finder,
        pairs=((gene, bundle.phenotype) for gene in positives),
    )
    return BenchmarkPool(
        candidates=bundle.candidate_pool,
        positives=positives,
        negatives=negatives,
        folds=temporal,
        leakage=audit.as_dict(),
    )


def cutoff_control_report(bundle: SubstrateBundle) -> dict[str, object]:
    _, audit = KnowledgeCutoffControl(bundle.spec.curation_cutoff).apply(bundle.graph)
    return audit.as_dict()


def pool_size_summary(bundle: SubstrateBundle) -> dict[str, object]:
    census = bundle.graph.census().as_dict()
    return {
        "candidates": len(bundle.candidate_pool),
        "graph_nodes": census["nodes"],
        "graph_edges": census["edges"],
        "metabolic_edges": census["metabolic_edges"],
        "held_out_candidates": len(bundle.held_out_genes),
    }
