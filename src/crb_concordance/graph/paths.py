"""Multi-hop retrieval of metabolic routes between a candidate and a phenotype.

Ref: Sec. 4.2 item (1) (path retrieval and grounding; pathway-level support
converted into m_g,KG); Sec. 4.4 (knowledge-cutoff and edge-holding controls).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.graph.substrate import Edge, KnowledgeGraph, NodeKind, Relation
from crb_concordance.utils.numerics import clip01

DEFAULT_MAX_HOPS = 3
HOP_DECAY = 0.55
PATHWAY_SATURATION = 4.0


class PathError(ValueError):
    """Raised when a path query is malformed or a node is absent."""


@dataclass(frozen=True, slots=True)
class GraphPath:
    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]

    @property
    def hops(self) -> int:
        return len(self.edges)

    def relations(self) -> tuple[str, ...]:
        return tuple(edge.relation.value for edge in self.edges)


@dataclass(frozen=True, slots=True)
class PathEvidence:
    """Retrieved support for one candidate with its full route provenance."""

    candidate: str
    phenotype: str
    paths: tuple[GraphPath, ...]
    pathway_hits: tuple[str, ...]
    support: float
    direct_edge_present: bool

    @property
    def hops(self) -> tuple[int, ...]:
        return tuple(path.hops for path in self.paths)

    def citations(self) -> tuple[str, ...]:
        return tuple(
            edge.evidence_ref for path in self.paths for edge in path.edges if edge.evidence_ref
        )


class PathFinder:
    """Bounded simple-path retrieval over a relation- and kind-filtered adjacency."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        *,
        max_hops: int = DEFAULT_MAX_HOPS,
        allowed_relations: tuple[Relation, ...] | None = None,
        allowed_kinds: tuple[NodeKind, ...] | None = None,
    ) -> None:
        if max_hops < 1:
            raise PathError(f"max_hops must be positive: {max_hops!r}")
        self.graph = graph
        self.max_hops = int(max_hops)
        self._relations = set(allowed_relations) if allowed_relations is not None else None
        self._kinds = set(allowed_kinds) if allowed_kinds is not None else None
        self._node_index = {key: position for position, key in enumerate(graph.node_keys)}
        self._keys = graph.node_keys
        self._indptr, self._targets, self._edge_ids = self._build_adjacency(graph, frozenset())

    def index_of(self, node: str) -> int:
        try:
            return self._node_index[node]
        except KeyError as error:
            raise PathError(f"unknown node: {node}") from error

    @property
    def allowed_relations(self) -> tuple[Relation, ...] | None:
        return (
            None
            if self._relations is None
            else tuple(sorted(self._relations, key=lambda r: r.value))
        )

    @property
    def allowed_kinds(self) -> tuple[NodeKind, ...] | None:
        return None if self._kinds is None else tuple(sorted(self._kinds, key=lambda k: k.value))

    def _accepts(self, edge: Edge) -> bool:
        if self._relations is not None and edge.relation not in self._relations:
            return False
        return self._kinds is None or self.graph.entity_kind(edge.dst) in self._kinds

    def _build_adjacency(
        self, graph: KnowledgeGraph, blocked_edges: frozenset[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = np.zeros(len(self._keys) + 1, dtype=np.int64)
        kept: list[tuple[int, int, int]] = []
        for position, edge in enumerate(graph.edges):
            if position in blocked_edges or not self._accepts(edge):
                continue
            source = self._node_index[edge.src]
            target = self._node_index[edge.dst]
            kept.append((source, target, position))
        for source, _, _ in kept:
            counts[source + 1] += 1
        indptr = np.cumsum(counts)
        targets = np.zeros(indptr[-1], dtype=np.int64)
        edge_ids = np.zeros(indptr[-1], dtype=np.int64)
        cursor = indptr[:-1].copy()
        for source, target, position in kept:
            targets[cursor[source]] = target
            edge_ids[cursor[source]] = position
            cursor[source] += 1
        return indptr, targets, edge_ids

    def _resolve(self, key: str, kind: NodeKind) -> str:
        return self.graph.resolve(key, kind)

    def depth_map(self, source: str) -> np.ndarray:
        """Vectorised breadth-first depth of every node within ``max_hops``."""

        source_index = self._node_index[source]
        depth = np.full(len(self._keys), -1, dtype=np.int32)
        depth[source_index] = 0
        frontier = np.array([source_index], dtype=np.int64)
        level = 0
        while frontier.size and level < self.max_hops:
            starts = self._indptr[frontier]
            ends = self._indptr[frontier + 1]
            counts = ends - starts
            total = int(counts.sum())
            if total == 0:
                break
            offsets = np.repeat(starts, counts) + (
                np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
            )
            discovered = np.unique(self._targets[offsets])
            fresh = discovered[depth[discovered] < 0]
            depth[fresh] = level + 1
            frontier = fresh
            level += 1
        return depth

    def enumerate_paths(
        self,
        candidate: str,
        phenotype: str,
        *,
        blocked_edges: frozenset[int] = frozenset(),
        max_hops: int | None = None,
    ) -> tuple[GraphPath, ...]:
        source = self._resolve(candidate, NodeKind.GENE)
        target = self._resolve(phenotype, NodeKind.PHENOTYPE)
        horizon = self.max_hops if max_hops is None else int(max_hops)
        if blocked_edges:
            indptr, targets, edge_ids = self._build_adjacency(self.graph, blocked_edges)
            graph_edges = self.graph.edges
        else:
            indptr, targets, edge_ids = self._indptr, self._targets, self._edge_ids
            graph_edges = self.graph.edges
        source_index = self._node_index[source]
        target_index = self._node_index[target]
        depth = self.depth_map(source)
        allowed = depth >= 0

        found: list[GraphPath] = []
        stack: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = [(source_index, (), ())]
        while stack:
            node, path_nodes, path_edges = stack.pop()
            if len(path_edges) >= horizon:
                continue
            for slot in range(int(indptr[node]), int(indptr[node + 1])):
                neighbour = int(targets[slot])
                if not allowed[neighbour] or neighbour in path_nodes or neighbour == source_index:
                    continue
                edge_position = int(edge_ids[slot])
                next_nodes = path_nodes + (neighbour,)
                next_edges = path_edges + (edge_position,)
                if neighbour == target_index:
                    found.append(
                        GraphPath(
                            nodes=(source,) + tuple(self._keys[i] for i in next_nodes),
                            edges=tuple(graph_edges[i] for i in next_edges),
                        )
                    )
                else:
                    stack.append((neighbour, next_nodes, next_edges))
        found.sort(key=lambda path: (path.hops, tuple(edge.relation.value for edge in path.edges)))
        return tuple(found)


def pathway_hits(graph: KnowledgeGraph, paths: tuple[GraphPath, ...]) -> tuple[str, ...]:
    """Distinct pathway nodes traversed by the retrieved routes."""

    seen: list[str] = []
    for path in paths:
        for node in path.nodes:
            if graph.entity_kind(node) is NodeKind.PATHWAY and node not in seen:
                seen.append(node)
    return tuple(seen)


def path_support(
    graph: KnowledgeGraph,
    paths: tuple[GraphPath, ...],
    *,
    decay: float = HOP_DECAY,
    saturation: float = PATHWAY_SATURATION,
) -> float:
    """Pathway-level support: decayed route mass saturating with pathway breadth."""

    if not paths:
        return 0.0
    hits = pathway_hits(graph, paths)
    breadth = len(hits) / saturation if saturation > 0 else float(len(hits))
    route_mass = 0.0
    for path in paths:
        weight = 1.0
        for edge in path.edges:
            weight *= max(edge.weight, 0.0)
        route_mass += decay**path.hops * weight
    route_term = route_mass / (1.0 + route_mass)
    return clip01(0.6 * min(1.0, breadth) + 0.4 * route_term)


def collect_path_evidence(
    finder: PathFinder,
    candidate: str,
    phenotype: str,
    *,
    blocked_edges: frozenset[int] = frozenset(),
) -> PathEvidence:
    paths = finder.enumerate_paths(
        candidate, phenotype, blocked_edges=blocked_edges, max_hops=finder.max_hops
    )
    source = finder.graph.resolve(candidate, NodeKind.GENE)
    target = finder.graph.resolve(phenotype, NodeKind.PHENOTYPE)
    direct = any(edge.dst == target for edge in finder.graph.out_edges(source))
    return PathEvidence(
        candidate=source,
        phenotype=target,
        paths=paths,
        pathway_hits=pathway_hits(finder.graph, paths),
        support=path_support(finder.graph, paths),
        direct_edge_present=direct,
    )
