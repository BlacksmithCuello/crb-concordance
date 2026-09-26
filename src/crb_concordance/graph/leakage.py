"""Leakage controls for the temporal-holdout rediscovery benchmark.

Ref: Sec. 4.4 (knowledge-cutoff control, substrate-circularity control and the
edge-holding scenario that forces indirect paths).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from crb_concordance.graph.paths import PathFinder
from crb_concordance.graph.substrate import Edge, KnowledgeGraph, NodeKind


class LeakageError(ValueError):
    """Raised when a control cannot be applied to the substrate as declared."""


@dataclass(frozen=True, slots=True)
class CutoffAudit:
    cutoff: str
    removed_edges: int
    undated_edges: int
    kept_edges: int
    removed_by_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "cutoff": self.cutoff,
            "removed_edges": self.removed_edges,
            "undated_edges": self.undated_edges,
            "kept_edges": self.kept_edges,
            "removed_by_source": dict(sorted(self.removed_by_source.items())),
        }


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


@dataclass(slots=True)
class KnowledgeCutoffControl:
    """Drop every edge curated after the declared evaluation cutoff.

    The timestamping unit is the source database release, because a curated graph
    does not timestamp individual edges.
    """

    cutoff: str
    require_dated: bool = False

    def apply(self, graph: KnowledgeGraph) -> tuple[KnowledgeGraph, CutoffAudit]:
        boundary = _parse_date(self.cutoff)
        if boundary is None:
            raise LeakageError(f"cutoff must be an ISO date: {self.cutoff!r}")

        def drop(edge: Edge) -> bool:
            released = _parse_date(edge.release_date)
            if released is None:
                return self.require_dated
            return released > boundary

        kept, removed = graph.filter_edges(drop)
        undated = sum(1 for edge in graph.edges if _parse_date(edge.release_date) is None)
        removed_by_source: dict[str, int] = {}
        for edge in graph.edges:
            released = _parse_date(edge.release_date)
            stale = released is None and self.require_dated
            if stale or (released is not None and released > boundary):
                removed_by_source[edge.source_db] = removed_by_source.get(edge.source_db, 0) + 1
        return kept, CutoffAudit(
            cutoff=self.cutoff,
            removed_edges=removed,
            undated_edges=undated,
            kept_edges=len(kept),
            removed_by_source=removed_by_source,
        )


@dataclass(frozen=True, slots=True)
class EdgeHoldoutAudit:
    candidate: str
    phenotype: str
    removed_edges: int
    blocked_positions: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "phenotype": self.phenotype,
            "removed_edges": self.removed_edges,
            "blocked_positions": list(self.blocked_positions),
        }


def edge_holdout(
    graph: KnowledgeGraph, candidate: str, phenotype: str
) -> tuple[KnowledgeGraph, EdgeHoldoutAudit]:
    """Remove the unique direct candidate-to-phenotype edge, forcing indirect routes."""

    source = graph.resolve(candidate, NodeKind.GENE)
    target = graph.resolve(phenotype, NodeKind.PHENOTYPE)
    positions = tuple(
        index for index, edge in enumerate(graph.edges) if edge.src == source and edge.dst == target
    )
    held, removed = graph.without_edge(source, target)
    audit = EdgeHoldoutAudit(
        candidate=source,
        phenotype=target,
        removed_edges=removed,
        blocked_positions=positions,
    )
    return held, audit


@dataclass(frozen=True, slots=True)
class CircularityAudit:
    reference: str
    removed_edges: int
    remaining_paths: int

    def as_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "removed_edges": self.removed_edges,
            "remaining_paths": self.remaining_paths,
        }


def substrate_circularity_control(
    graph: KnowledgeGraph,
    finder: PathFinder,
    candidate: str,
    phenotype: str,
    *,
    reference: str,
) -> tuple[KnowledgeGraph, CircularityAudit]:
    """Remove edges whose only support is the positive-control publication."""

    keep, removed = graph.filter_edges(
        lambda edge: bool(reference) and reference in edge.evidence_ref
    )
    trimmed = PathFinder(
        keep,
        max_hops=finder.max_hops,
        allowed_relations=finder.allowed_relations,
        allowed_kinds=finder.allowed_kinds,
    )
    remaining = len(trimmed.enumerate_paths(candidate, phenotype))
    return keep, CircularityAudit(
        reference=reference, removed_edges=removed, remaining_paths=remaining
    )


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    cutoff: CutoffAudit
    circularity: CircularityAudit
    held_out_direct_edges: tuple[EdgeHoldoutAudit, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "cutoff": self.cutoff.as_dict(),
            "circularity": self.circularity.as_dict(),
            "held_out_direct_edges": [audit.as_dict() for audit in self.held_out_direct_edges],
        }


def leakage_audit(
    graph: KnowledgeGraph,
    *,
    cutoff: str,
    reference: str,
    finder: PathFinder,
    pairs: Iterable[tuple[str, str]],
) -> LeakageAudit:
    """Run all three controls and report them together, as the benchmark requires."""

    _, cutoff_audit = KnowledgeCutoffControl(cutoff).apply(graph)
    first_candidate, first_phenotype = next(iter(pairs))
    _, circularity = substrate_circularity_control(
        graph, finder, first_candidate, first_phenotype, reference=reference
    )
    holdouts = tuple(edge_holdout(graph, candidate, phenotype)[1] for candidate, phenotype in pairs)
    return LeakageAudit(
        cutoff=cutoff_audit, circularity=circularity, held_out_direct_edges=holdouts
    )
