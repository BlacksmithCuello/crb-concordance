"""Attaching genome-scale metabolic reaction edges to the substrate.

Ref: Sec. 4.3 (the substrate is extended with metabolic reactions sourced from
Human-GEM v2.0.1); Sec. 4.4 (these merging edges are structurally new relative to
any knowledge-graph foundation model's pretraining graph, so they are the
generalization axis of the benchmark).
"""

from __future__ import annotations

from crb_concordance.graph.substrate import Edge, Entity, KnowledgeGraph, NodeKind, Relation
from crb_concordance.metabolism.reactions import MetabolicReaction

HUMAN_GEM_SOURCE = "Human-GEM v2.0.1"


class MetabolicEdgeError(ValueError):
    """Raised when a reaction references a node the substrate cannot register."""


def _ensure(graph: KnowledgeGraph, key: str, kind: NodeKind, name: str) -> None:
    if key in graph.node_keys:
        return
    graph.add_entity(Entity(key=key, kind=kind, name=name, identifiers=(("key", key),)))


def attach_reaction_edges(
    graph: KnowledgeGraph,
    reactions: tuple[MetabolicReaction, ...],
    *,
    source_db: str = HUMAN_GEM_SOURCE,
    release_date: str,
    edge_weight: float = 1.0,
) -> int:
    """Register reaction, metabolite and pathway nodes, then wire the reaction edges."""

    added = 0
    for reaction in reactions:
        reaction.validate()
        _ensure(graph, reaction.pathway, NodeKind.PATHWAY, reaction.pathway)
        for gene in reaction.enzymes:
            _ensure(graph, gene, NodeKind.GENE, gene)
        for metabolite in set(reaction.substrates) | set(reaction.products):
            _ensure(graph, metabolite, NodeKind.METABOLITE, metabolite)
        _ensure(graph, reaction.reaction_id, NodeKind.REACTION, reaction.name)
        for gene in reaction.enzymes:
            graph.add_edge(
                Edge(
                    src=gene,
                    dst=reaction.reaction_id,
                    relation=Relation.REACTION_ENZYME,
                    source_db=source_db,
                    release_date=release_date,
                    weight=edge_weight,
                    evidence_ref=f"{source_db}/{reaction.reaction_id}/gpr",
                )
            )
            graph.add_edge(
                Edge(
                    src=gene,
                    dst=reaction.pathway,
                    relation=Relation.GENE_PATHWAY_MEMBERSHIP,
                    source_db=source_db,
                    release_date=release_date,
                    weight=edge_weight,
                    evidence_ref=f"{source_db}/{reaction.reaction_id}/subsystem",
                )
            )
            added += 2
        for metabolite in reaction.substrates:
            graph.add_edge(
                Edge(
                    src=metabolite,
                    dst=reaction.reaction_id,
                    relation=Relation.METABOLIC_REACTION,
                    source_db=source_db,
                    release_date=release_date,
                    weight=edge_weight,
                    evidence_ref=f"{source_db}/{reaction.reaction_id}/substrate",
                )
            )
            added += 1
        for metabolite in reaction.products:
            graph.add_edge(
                Edge(
                    src=reaction.reaction_id,
                    dst=metabolite,
                    relation=Relation.METABOLIC_REACTION,
                    source_db=source_db,
                    release_date=release_date,
                    weight=edge_weight,
                    evidence_ref=f"{source_db}/{reaction.reaction_id}/product",
                )
            )
            added += 1
        if reaction.reversible:
            for metabolite in reaction.products:
                graph.add_edge(
                    Edge(
                        src=metabolite,
                        dst=reaction.reaction_id,
                        relation=Relation.METABOLIC_REACTION,
                        source_db=source_db,
                        release_date=release_date,
                        weight=edge_weight,
                        evidence_ref=f"{source_db}/{reaction.reaction_id}/reverse",
                    )
                )
                added += 1
        for metabolite in reaction.substrates:
            graph.add_edge(
                Edge(
                    src=metabolite,
                    dst=reaction.pathway,
                    relation=Relation.METABOLITE_PATHWAY,
                    source_db=source_db,
                    release_date=release_date,
                    weight=edge_weight,
                    evidence_ref=f"{source_db}/{reaction.reaction_id}/metabolite-pathway",
                )
            )
            added += 1
    return added


def metabolic_edge_count(graph: KnowledgeGraph) -> int:
    relations = (
        Relation.METABOLIC_REACTION,
        Relation.REACTION_ENZYME,
        Relation.METABOLITE_PATHWAY,
    )
    return sum(1 for edge in graph.edges if edge.relation in relations)


def enzyme_partners(graph: KnowledgeGraph, reaction_id: str) -> tuple[str, ...]:
    return tuple(
        edge.src
        for edge in graph.in_edges(reaction_id)
        if edge.relation is Relation.REACTION_ENZYME
    )


def pathway_of(graph: KnowledgeGraph, reaction_id: str) -> str | None:
    """The pathway a reaction belongs to, resolved through its enzyme genes.

    The metabolic layer attaches pathway membership to the gating gene rather than to
    the reaction, so the route runs reaction -> enzyme -> pathway.
    """

    for edge in graph.in_edges(reaction_id):
        if edge.relation is Relation.GENE_PATHWAY_MEMBERSHIP:
            return edge.dst
    for enzyme in enzyme_partners(graph, reaction_id):
        for edge in graph.out_edges(enzyme):
            if edge.relation is Relation.GENE_PATHWAY_MEMBERSHIP:
                return edge.dst
    return None
