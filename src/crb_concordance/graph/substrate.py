"""Typed biomedical substrate: entities, relations and release-dated edges.

Ref: Sec. 4.3 (auxiliary public layer: the curated graph cross-validated against
Hetionet v1.0 and extended with Human-GEM v2.0.1 reaction edges); Sec. 4.4
(the release date of the source database is the timestamping unit).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class NodeKind(str, Enum):
    GENE = "gene"
    PROTEIN = "protein"
    METABOLITE = "metabolite"
    REACTION = "reaction"
    PATHWAY = "pathway"
    DISEASE = "disease"
    PHENOTYPE = "phenotype"
    DRUG = "drug"
    CELL_LINE = "cell_line"


class Relation(str, Enum):
    GENE_DISEASE_ASSOCIATION = "gene_disease_association"
    DRUG_TARGET = "drug_target"
    DRUG_DISEASE_TREATMENT = "drug_disease_treatment"
    GENE_PATHWAY_MEMBERSHIP = "gene_pathway_membership"
    PATHWAY_DISEASE_ASSOCIATION = "pathway_disease_association"
    METABOLIC_REACTION = "metabolic_reaction"
    REACTION_ENZYME = "reaction_enzyme"
    METABOLITE_PATHWAY = "metabolite_pathway"
    GENE_PHENOTYPE_ASSOCIATION = "gene_phenotype_association"
    TRANSCRIPTION_REGULATION = "transcription_regulation"
    PROTEIN_INTERACTION = "protein_interaction"


class GraphError(ValueError):
    """Raised on malformed substrate input or an unknown node reference."""


@dataclass(frozen=True, slots=True)
class Entity:
    key: str
    kind: NodeKind
    name: str
    identifiers: tuple[tuple[str, str], ...] = ()

    def identifier(self, namespace: str) -> str | None:
        for name, value in self.identifiers:
            if name == namespace:
                return value
        return None


@dataclass(frozen=True, slots=True)
class Edge:
    src: str
    dst: str
    relation: Relation
    source_db: str
    release_date: str
    weight: float = 1.0
    evidence_ref: str = ""

    def key(self) -> tuple[str, str, str]:
        return (self.src, self.dst, self.relation.value)


@dataclass(slots=True)
class GraphCensus:
    nodes: int = 0
    edges: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    by_relation: dict[str, int] = field(default_factory=dict)
    by_source_db: dict[str, int] = field(default_factory=dict)
    metabolic_edges: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "by_kind": dict(sorted(self.by_kind.items())),
            "by_relation": dict(sorted(self.by_relation.items())),
            "by_source_db": dict(sorted(self.by_source_db.items())),
            "metabolic_edges": self.metabolic_edges,
        }


class KnowledgeGraph:
    """Directed multi-relation graph with per-edge provenance and release dates."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[int]] = defaultdict(list)
        self._in: dict[str, list[int]] = defaultdict(list)
        self._index: dict[tuple[str, str, str], int] = {}

    def __len__(self) -> int:
        return len(self._edges)

    def __iter__(self) -> Iterator[Edge]:
        return iter(self._edges)

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges)

    @property
    def entities(self) -> tuple[Entity, ...]:
        return tuple(self._entities.values())

    @property
    def node_keys(self) -> tuple[str, ...]:
        return tuple(self._entities)

    def clone(self) -> KnowledgeGraph:
        """A shallow copy that keeps nodes and edges in insertion order."""

        duplicate = KnowledgeGraph()
        for entity in self._entities.values():
            duplicate.add_entity(entity)
        for edge in self._edges:
            duplicate.add_edge(edge)
        return duplicate

    def filter_edges(self, drop: Callable[[Edge], bool]) -> tuple[KnowledgeGraph, int]:
        """Copy the graph without the edges selected by ``drop``."""

        kept = KnowledgeGraph()
        for entity in self._entities.values():
            kept.add_entity(entity)
        removed = 0
        for edge in self._edges:
            if drop(edge):
                removed += 1
                continue
            kept.add_edge(edge)
        return kept, removed

    def add_entity(self, entity: Entity) -> None:
        existing = self._entities.get(entity.key)
        if existing is None:
            self._entities[entity.key] = entity
        elif existing.kind != entity.kind:
            raise GraphError(
                f"conflicting kinds for {entity.key}: {existing.kind} vs {entity.kind}"
            )

    def resolve(self, key: str, kind: NodeKind | None = None) -> str:
        """Resolve a free-form identifier to a node key, honouring a kind filter."""

        if key in self._entities:
            entity = self._entities[key]
            if kind is not None and entity.kind != kind:
                raise GraphError(f"{key} is a {entity.kind.value}, expected {kind.value}")
            return key
        for candidate, entity in self._entities.items():
            if kind is not None and entity.kind != kind:
                continue
            for namespace, value in entity.identifiers:
                if value.upper() == key.upper() or namespace.upper() == key.upper():
                    return candidate
        raise GraphError(f"unknown node: {key}")

    def add_edge(self, edge: Edge) -> None:
        for node in (edge.src, edge.dst):
            if node not in self._entities:
                raise GraphError(f"edge references unknown node: {node}")
        edge_key = edge.key()
        if edge_key in self._index:
            return
        self._index[edge_key] = len(self._edges)
        self._edges.append(edge)
        self._out[edge.src].append(len(self._edges) - 1)
        self._in[edge.dst].append(len(self._edges) - 1)

    def out_edges(self, node: str) -> tuple[Edge, ...]:
        return tuple(self._edges[i] for i in self._out.get(node, ()))

    def in_edges(self, node: str) -> tuple[Edge, ...]:
        return tuple(self._edges[i] for i in self._in.get(node, ()))

    def neighbours(self, node: str) -> tuple[str, ...]:
        return tuple(self._edges[i].dst for i in self._out.get(node, ()))

    def has_edge(self, src: str, dst: str, relation: Relation | None = None) -> bool:
        return any(
            edge.relation == relation if relation is not None else True
            for edge in self.out_edges(src)
            if edge.dst == dst
        )

    def entity_kind(self, node: str) -> NodeKind:
        return self._entities[node].kind

    def census(self) -> GraphCensus:
        by_kind: dict[str, int] = defaultdict(int)
        for entity in self._entities.values():
            by_kind[entity.kind.value] += 1
        by_relation: dict[str, int] = defaultdict(int)
        by_source: dict[str, int] = defaultdict(int)
        metabolic = 0
        for edge in self._edges:
            by_relation[edge.relation.value] += 1
            by_source[edge.source_db] += 1
            if edge.relation in (
                Relation.METABOLIC_REACTION,
                Relation.REACTION_ENZYME,
                Relation.METABOLITE_PATHWAY,
            ):
                metabolic += 1
        return GraphCensus(
            nodes=len(self._entities),
            edges=len(self._edges),
            by_kind=dict(by_kind),
            by_relation=dict(by_relation),
            by_source_db=dict(by_source),
            metabolic_edges=metabolic,
        )

    def induced(self, keep: Iterable[str]) -> KnowledgeGraph:
        wanted = set(keep)
        subgraph = KnowledgeGraph()
        for key in wanted:
            if key in self._entities:
                subgraph.add_entity(self._entities[key])
        for edge in self._edges:
            if edge.src in wanted and edge.dst in wanted:
                subgraph.add_edge(edge)
        return subgraph

    def without_edge(self, src: str, dst: str) -> tuple[KnowledgeGraph, int]:
        """Return a copy without any src -> dst edge, plus the number removed."""

        dropped = 0
        kept = KnowledgeGraph()
        for entity in self._entities.values():
            kept.add_entity(entity)
        for edge in self._edges:
            if edge.src == src and edge.dst == dst:
                dropped += 1
                continue
            kept.add_edge(edge)
        return kept, dropped

    def to_records(self) -> dict[str, object]:
        return {
            "entities": [
                {
                    "key": entity.key,
                    "kind": entity.kind.value,
                    "name": entity.name,
                    "identifiers": [list(pair) for pair in entity.identifiers],
                }
                for entity in self._entities.values()
            ],
            "edges": [
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "relation": edge.relation.value,
                    "source_db": edge.source_db,
                    "release_date": edge.release_date,
                    "weight": edge.weight,
                    "evidence_ref": edge.evidence_ref,
                }
                for edge in self._edges
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_records(), indent=1), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        graph = cls()
        for record in payload["entities"]:
            graph.add_entity(
                Entity(
                    key=record["key"],
                    kind=NodeKind(record["kind"]),
                    name=record["name"],
                    identifiers=tuple(tuple(pair) for pair in record.get("identifiers", ())),
                )
            )
        for record in payload["edges"]:
            graph.add_edge(
                Edge(
                    src=record["src"],
                    dst=record["dst"],
                    relation=Relation(record["relation"]),
                    source_db=record["source_db"],
                    release_date=record["release_date"],
                    weight=float(record["weight"]),
                    evidence_ref=record.get("evidence_ref", ""),
                )
            )
        return graph
