"""Substrate, path retrieval and the leakage controls."""

from __future__ import annotations

import pytest

from crb_concordance.graph.leakage import (
    KnowledgeCutoffControl,
    LeakageError,
    edge_holdout,
    leakage_audit,
    substrate_circularity_control,
)
from crb_concordance.graph.metabolic_edges import (
    HUMAN_GEM_SOURCE,
    attach_reaction_edges,
    enzyme_partners,
    metabolic_edge_count,
    pathway_of,
)
from crb_concordance.graph.paths import (
    PathError,
    PathFinder,
    collect_path_evidence,
    path_support,
    pathway_hits,
)
from crb_concordance.graph.substrate import (
    Edge,
    Entity,
    GraphError,
    KnowledgeGraph,
    NodeKind,
    Relation,
)
from crb_concordance.metabolism.catalogue import build_catalogue

CONTROL_REFERENCE = "Hellkamp 2026"


def _toy_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for key, kind in (
        ("A", NodeKind.GENE),
        ("B", NodeKind.GENE),
        ("C", NodeKind.GENE),
        ("P1", NodeKind.PATHWAY),
        ("P2", NodeKind.PATHWAY),
        ("PH", NodeKind.PHENOTYPE),
    ):
        graph.add_entity(Entity(key=key, kind=kind, name=key))
    for source, target in (("A", "P1"), ("P1", "PH"), ("B", "P1"), ("C", "P2"), ("P2", "PH")):
        graph.add_edge(
            Edge(
                src=source,
                dst=target,
                relation=Relation.PATHWAY_DISEASE_ASSOCIATION,
                source_db="test",
                release_date="2024-01-01",
                weight=1.0,
            )
        )
    return graph


def test_toy_graph_enumerates_the_expected_routes() -> None:
    finder = PathFinder(
        _toy_graph(), max_hops=3, allowed_kinds=(NodeKind.PATHWAY, NodeKind.PHENOTYPE)
    )
    a_routes = finder.enumerate_paths("A", "PH")
    c_routes = finder.enumerate_paths("C", "PH")
    assert len(a_routes) == 1
    assert a_routes[0].nodes == ("A", "P1", "PH")
    assert a_routes[0].hops == 2
    assert len(c_routes) == 1


def test_unreachable_candidate_reports_no_route() -> None:
    graph = _toy_graph()
    graph.add_entity(Entity(key="Z", kind=NodeKind.GENE, name="Z"))
    finder = PathFinder(graph, max_hops=3, allowed_kinds=(NodeKind.PATHWAY, NodeKind.PHENOTYPE))
    assert finder.enumerate_paths("Z", "PH") == ()


def test_depth_map_respects_the_hop_bound() -> None:
    finder = PathFinder(
        _toy_graph(), max_hops=1, allowed_kinds=(NodeKind.PATHWAY, NodeKind.PHENOTYPE)
    )
    depth = finder.depth_map("A")
    assert depth[finder._node_index["A"]] == 0
    assert depth[finder._node_index["P1"]] == 1
    assert depth[finder._node_index["PH"]] == -1


def test_path_retrieval_on_the_substrate_is_simple_and_bounded(substrate) -> None:
    finder = PathFinder(substrate.graph, max_hops=3)
    for gene in ("SLC2A1", "LDHA", "G6PD"):
        routes = finder.enumerate_paths(gene, substrate.phenotype)
        for route in routes:
            assert len(set(route.nodes)) == len(route.nodes)
            assert route.hops <= 3
            assert route.nodes[0] == gene
            assert route.nodes[-1] == substrate.phenotype


def test_path_support_is_bounded_and_uses_pathways(substrate) -> None:
    finder = PathFinder(substrate.graph, max_hops=3)
    routes = finder.enumerate_paths("SLC2A1", substrate.phenotype)
    hits = pathway_hits(substrate.graph, routes)
    support = path_support(substrate.graph, routes)
    assert 0.0 <= support <= 1.0
    assert support > 0.0
    assert hits
    assert path_support(substrate.graph, ()) == 0.0


def test_collect_path_evidence_reports_the_direct_edge(substrate) -> None:
    finder = PathFinder(substrate.graph, max_hops=3)
    evidence = collect_path_evidence(finder, "SLC2A1", substrate.phenotype)
    assert evidence.direct_edge_present
    assert evidence.candidate == "SLC2A1"
    assert evidence.hops


def test_hop_bound_is_enforced() -> None:
    with pytest.raises(PathError):
        PathFinder(_toy_graph(), max_hops=0)


def test_unknown_node_raises() -> None:
    finder = PathFinder(_toy_graph(), max_hops=2)
    with pytest.raises(GraphError):
        finder.enumerate_paths("NOT_THERE", "PH")


def test_substrate_census_counts_the_declared_sources(substrate) -> None:
    census = substrate.graph.census()
    assert census.nodes > 0
    assert census.edges > 0
    assert census.metabolic_edges > 0
    assert set(census.by_source_db) >= {
        "Human-GEM v2.0.1",
        "OptimusKG",
        "Reactome",
        "KEGG colorectal cancer pathway map",
    }


def test_cutoff_control_removes_only_post_cutoff_edges(substrate) -> None:
    control = KnowledgeCutoffControl(substrate.spec.curation_cutoff)
    trimmed, audit = control.apply(substrate.graph)
    assert audit.removed_edges > 0
    assert audit.kept_edges == len(trimmed)
    assert all(
        edge.release_date <= substrate.spec.curation_cutoff
        for edge in trimmed.edges
        if edge.release_date
    )
    assert sum(audit.removed_by_source.values()) == audit.removed_edges


def test_cutoff_control_demands_an_iso_date(substrate) -> None:
    with pytest.raises(LeakageError):
        KnowledgeCutoffControl("mid 2026").apply(substrate.graph)


def test_edge_holdout_forces_indirect_routes(substrate) -> None:
    held, audit = edge_holdout(substrate.graph, "SLC2A1", substrate.phenotype)
    assert audit.removed_edges >= 1
    finder = PathFinder(held, max_hops=3)
    assert finder.enumerate_paths("SLC2A1", substrate.phenotype)
    assert not any(edge.src == "SLC2A1" and edge.dst == substrate.phenotype for edge in held.edges)


def test_circularity_control_strips_the_positive_control_reference(substrate) -> None:
    finder = PathFinder(substrate.graph, max_hops=3)
    before = len(finder.enumerate_paths("SLC2A1", substrate.phenotype))
    trimmed, audit = substrate_circularity_control(
        substrate.graph, finder, "SLC2A1", substrate.phenotype, reference=CONTROL_REFERENCE
    )
    assert audit.removed_edges > 0
    assert len(trimmed) < len(substrate.graph)
    assert before >= audit.remaining_paths


def test_leakage_audit_reports_all_three_controls(substrate) -> None:
    finder = PathFinder(substrate.graph, max_hops=3)
    audit = leakage_audit(
        substrate.graph,
        cutoff=substrate.spec.curation_cutoff,
        reference=CONTROL_REFERENCE,
        finder=finder,
        pairs=(("SLC2A1", substrate.phenotype), ("LDHA", substrate.phenotype)),
    )
    payload = audit.as_dict()
    assert set(payload) == {"cutoff", "circularity", "held_out_direct_edges"}
    assert len(payload["held_out_direct_edges"]) == 2


def test_metabolic_edges_carry_the_human_gem_source() -> None:
    graph = KnowledgeGraph()
    graph.add_entity(Entity(key="G1", kind=NodeKind.GENE, name="G1"))
    attached = attach_reaction_edges(
        graph, tuple(entry.reaction for entry in build_catalogue()), release_date="2023-06-30"
    )
    assert attached > 0
    assert metabolic_edge_count(graph) > 0
    assert all(
        edge.source_db == HUMAN_GEM_SOURCE
        for edge in graph.edges
        if edge.relation is Relation.METABOLIC_REACTION
    )
    assert enzyme_partners(graph, "HEX") != ()
    assert pathway_of(graph, "HEX") is not None


def test_duplicate_edges_are_ignored() -> None:
    graph = _toy_graph()
    before = len(graph)
    graph.add_edge(
        Edge(
            src="A",
            dst="P1",
            relation=Relation.PATHWAY_DISEASE_ASSOCIATION,
            source_db="test",
            release_date="2024-01-01",
        )
    )
    assert len(graph) == before


def test_edge_to_unknown_node_is_rejected() -> None:
    graph = _toy_graph()
    with pytest.raises(GraphError):
        graph.add_edge(
            Edge(
                src="A",
                dst="NOPE",
                relation=Relation.GENE_PATHWAY_MEMBERSHIP,
                source_db="test",
                release_date="2024-01-01",
            )
        )


def test_graph_round_trips_through_json(tmp_path, substrate) -> None:
    target = substrate.graph.save(tmp_path / "graph.json")
    restored = KnowledgeGraph.load(target)
    assert len(restored) == len(substrate.graph)
    assert restored.census().as_dict() == substrate.graph.census().as_dict()
