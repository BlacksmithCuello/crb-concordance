"""Schema-compatible construction of the knowledge-graph substrate.

Ref: Sec. 4.3 (the substrate is a curated biomedical graph cross-validated against
Hetionet v1.0 and extended with Human-GEM v2.0.1 metabolic reactions, with Reactome
and the KEGG colorectal cancer pathway map supplying pathway context);
Sec. 4.4 (every edge carries the release date of its source database, and the
direct candidate-to-phenotype edge is the unit the edge-holding control removes).

The OptimusKG deposit is CC BY-NC-SA 4.0 and is not redistributed here, so this
module builds a substrate that carries the declared relation types, node kinds,
release-date semantics and the named pathway structure, which is what the retrieval
and leakage controls operate on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crb_concordance.graph.metabolic_edges import attach_reaction_edges
from crb_concordance.graph.substrate import Edge, Entity, KnowledgeGraph, NodeKind, Relation
from crb_concordance.metabolism.catalogue import (
    FERROPTOSIS,
    GLYCOLYSIS,
    NEGATIVE_CONTROLS,
    NON_GENE_MARKERS,
    NUCLEOTIDE_REPAIR,
    OXPHOS,
    POSITIVE_CONTROLS,
    build_catalogue,
    reaction_genes,
)
from crb_concordance.metabolism.reactions import MetabolicReaction
from crb_concordance.utils.numerics import rng_from

PHENOTYPE_KEY = "radioresistance"
DISEASE_KEY = "colorectal_adenocarcinoma"
OPTIMUSKG_SOURCE = "OptimusKG"
HETIONET_SOURCE = "Hetionet v1.0"
REACTOME_SOURCE = "Reactome"
KEGG_SOURCE = "KEGG colorectal cancer pathway map"
HUMAN_GEM_RELEASE_DATE = "2023-06-30"
POSITIVE_CONTROL_REFERENCE = "Hellkamp 2026 lactate axis screen"

PATHWAY_NAMES: tuple[str, ...] = (
    GLYCOLYSIS,
    OXPHOS,
    FERROPTOSIS,
    NUCLEOTIDE_REPAIR,
    "tca-cycle",
    "pentose-phosphate",
    "one-carbon",
    "glutaminolysis",
    "fatty-acid-oxidation-carnitine-shuttle",
    "fatty-acid-synthesis",
    "redox-shuttle",
    "transport",
)


@dataclass(frozen=True, slots=True)
class SubstrateSpec:
    """Sizing and release calendar of the substrate stand-in."""

    n_background_genes: int = 72
    n_drugs: int = 24
    seed: int = 20260101
    pre_cutoff_date: str = "2025-06-30"
    post_cutoff_date: str = "2026-05-31"
    curation_cutoff: str = "2026-01-01"

    def as_dict(self) -> dict[str, object]:
        return {
            "n_background_genes": self.n_background_genes,
            "n_drugs": self.n_drugs,
            "seed": self.seed,
            "pre_cutoff_date": self.pre_cutoff_date,
            "post_cutoff_date": self.post_cutoff_date,
            "curation_cutoff": self.curation_cutoff,
        }


@dataclass(frozen=True, slots=True)
class SubstrateBundle:
    graph: KnowledgeGraph
    phenotype: str
    disease: str
    candidate_pool: tuple[str, ...]
    held_out_genes: tuple[str, ...]
    spec: SubstrateSpec

    def as_dict(self) -> dict[str, object]:
        census = self.graph.census().as_dict()
        return {
            "spec": self.spec.as_dict(),
            "phenotype": self.phenotype,
            "disease": self.disease,
            "candidates": len(self.candidate_pool),
            "held_out_genes": len(self.held_out_genes),
            "census": census,
        }


def _add_phenotype_nodes(graph: KnowledgeGraph) -> None:
    graph.add_entity(
        Entity(
            key=PHENOTYPE_KEY,
            kind=NodeKind.PHENOTYPE,
            name="radiotherapy resistance",
            identifiers=(("key", "RADIORESISTANCE"),),
        )
    )
    graph.add_entity(
        Entity(
            key=DISEASE_KEY,
            kind=NodeKind.DISEASE,
            name="colorectal adenocarcinoma",
            identifiers=(("TCGA", "COAD"), ("TCGA", "READ")),
        )
    )


def _add_gene(graph: KnowledgeGraph, symbol: str, alias: str | None = None) -> None:
    identifiers: list[tuple[str, str]] = [("symbol", symbol)]
    if alias is not None:
        identifiers.append(("alias", alias))
    graph.add_entity(
        Entity(
            key=symbol,
            kind=NodeKind.GENE,
            name=f"{symbol} gene",
            identifiers=tuple(identifiers),
        )
    )


def _add_drug(graph: KnowledgeGraph, symbol: str) -> None:
    graph.add_entity(
        Entity(key=symbol, kind=NodeKind.DRUG, name=symbol, identifiers=(("key", symbol),))
    )


def build_substrate(spec: SubstrateSpec | None = None) -> SubstrateBundle:
    """Assemble the substrate with release-dated edges and a declared holdout."""

    settings = spec or SubstrateSpec()
    graph = KnowledgeGraph()
    _add_phenotype_nodes(graph)

    for symbol in reaction_genes(build_catalogue()):
        _add_gene(graph, symbol)
    for symbol in POSITIVE_CONTROLS + NEGATIVE_CONTROLS:
        _add_gene(graph, symbol)
    for position in range(settings.n_background_genes):
        _add_gene(graph, f"BG{position:03d}")

    for pathway in PATHWAY_NAMES:
        graph.add_entity(
            Entity(
                key=pathway, kind=NodeKind.PATHWAY, name=pathway, identifiers=(("key", pathway),)
            )
        )
    for position in range(settings.n_drugs):
        _add_drug(graph, f"DRUG{position:03d}")

    reaction_edges = attach_reaction_edges(
        graph,
        build_catalogue_reactions(),
        release_date=HUMAN_GEM_RELEASE_DATE,
    )
    if reaction_edges == 0:
        raise RuntimeError("the metabolic catalogue produced no reaction edges")

    genes = tuple(entity.key for entity in graph.entities if entity.kind is NodeKind.GENE)
    for position, gene in enumerate(genes):
        pathway = PATHWAY_NAMES[position % len(PATHWAY_NAMES)]
        graph.add_edge(
            Edge(
                src=gene,
                dst=pathway,
                relation=Relation.GENE_PATHWAY_MEMBERSHIP,
                source_db=REACTOME_SOURCE,
                release_date="2024-12-31",
                weight=1.0,
                evidence_ref=f"{REACTOME_SOURCE}/{pathway}/membership",
            )
        )
        if position % 5 == 0:
            graph.add_edge(
                Edge(
                    src=gene,
                    dst=DISEASE_KEY,
                    relation=Relation.GENE_DISEASE_ASSOCIATION,
                    source_db=OPTIMUSKG_SOURCE,
                    release_date=(
                        settings.post_cutoff_date
                        if position % 25 == 0
                        else settings.pre_cutoff_date
                    ),
                    weight=0.8,
                    evidence_ref=f"{OPTIMUSKG_SOURCE}/{gene}/disease",
                )
            )
        if position % 7 == 0:
            partner = genes[(position * 3 + 5) % len(genes)]
            if partner != gene:
                graph.add_edge(
                    Edge(
                        src=gene,
                        dst=partner,
                        relation=Relation.PROTEIN_INTERACTION,
                        source_db=HETIONET_SOURCE,
                        release_date=settings.pre_cutoff_date,
                        weight=0.6,
                        evidence_ref=f"{HETIONET_SOURCE}/{gene}-{partner}",
                    )
                )
        if position % 11 == 0:
            drug = f"DRUG{position % settings.n_drugs:03d}"
            graph.add_edge(
                Edge(
                    src=drug,
                    dst=gene,
                    relation=Relation.DRUG_TARGET,
                    source_db=OPTIMUSKG_SOURCE,
                    release_date=settings.pre_cutoff_date,
                    weight=0.7,
                    evidence_ref=f"{OPTIMUSKG_SOURCE}/{drug}/target",
                )
            )

    for pathway in PATHWAY_NAMES:
        graph.add_edge(
            Edge(
                src=pathway,
                dst=DISEASE_KEY,
                relation=Relation.PATHWAY_DISEASE_ASSOCIATION,
                source_db=KEGG_SOURCE,
                release_date="2024-05-31",
                weight=0.9,
                evidence_ref=f"{KEGG_SOURCE}/{pathway}",
            )
        )
        graph.add_edge(
            Edge(
                src=pathway,
                dst=PHENOTYPE_KEY,
                relation=Relation.PATHWAY_DISEASE_ASSOCIATION,
                source_db=KEGG_SOURCE,
                release_date="2024-05-31",
                weight=0.7,
                evidence_ref=f"{KEGG_SOURCE}/{pathway}/radioresistance",
            )
        )

    for drug in (f"DRUG{index:03d}" for index in range(settings.n_drugs)):
        graph.add_edge(
            Edge(
                src=drug,
                dst=DISEASE_KEY,
                relation=Relation.DRUG_DISEASE_TREATMENT,
                source_db=OPTIMUSKG_SOURCE,
                release_date=settings.pre_cutoff_date,
                weight=0.5,
                evidence_ref=f"{OPTIMUSKG_SOURCE}/{drug}/indication",
            )
        )

    held_out: list[str] = []
    for position, gene in enumerate(genes):
        post_cutoff = position % 9 == 0
        reference = (
            POSITIVE_CONTROL_REFERENCE
            if gene in POSITIVE_CONTROLS
            else f"{OPTIMUSKG_SOURCE}/{gene}/phenotype"
        )
        graph.add_edge(
            Edge(
                src=gene,
                dst=PHENOTYPE_KEY,
                relation=Relation.GENE_PHENOTYPE_ASSOCIATION,
                source_db=OPTIMUSKG_SOURCE,
                release_date=(
                    settings.post_cutoff_date if post_cutoff else settings.pre_cutoff_date
                ),
                weight=0.75,
                evidence_ref=reference,
            )
        )
        if post_cutoff:
            held_out.append(gene)

    pool = tuple(
        sorted(
            entity.key
            for entity in graph.entities
            if entity.kind is NodeKind.GENE and entity.key not in NON_GENE_MARKERS
        )
    )
    return SubstrateBundle(
        graph=graph,
        phenotype=PHENOTYPE_KEY,
        disease=DISEASE_KEY,
        candidate_pool=pool,
        held_out_genes=tuple(sorted(held_out)),
        spec=settings,
    )


def build_catalogue_reactions() -> tuple[MetabolicReaction, ...]:
    return tuple(entry.reaction for entry in build_catalogue())


def save_bundle(bundle: SubstrateBundle, path: str | Path) -> Path:
    return bundle.graph.save(path)


def load_bundle(path: str | Path, spec: SubstrateSpec | None = None) -> SubstrateBundle:
    graph = KnowledgeGraph.load(path)
    settings = spec or SubstrateSpec()
    pool = tuple(sorted(entity.key for entity in graph.entities if entity.kind is NodeKind.GENE))
    held_out = tuple(
        sorted(
            edge.src
            for edge in graph.edges
            if edge.dst == PHENOTYPE_KEY and edge.release_date > settings.curation_cutoff
        )
    )
    return SubstrateBundle(
        graph=graph,
        phenotype=PHENOTYPE_KEY,
        disease=DISEASE_KEY,
        candidate_pool=pool,
        held_out_genes=held_out,
        spec=settings,
    )


def dependency_table_from_substrate(
    bundle: SubstrateBundle,
) -> dict[str, tuple[float, float | None]]:
    """Chronos-like gene effects consistent with the substrate's own holdout marking."""

    generator = rng_from(bundle.spec.seed + 7)
    table: dict[str, tuple[float, float | None]] = {}
    for gene in bundle.candidate_pool:
        effect = float(generator.normal(-0.25, 0.45))
        response: float | None = float(generator.normal(-0.4, 0.6))
        if generator.random() < 0.18:
            response = None
        table[gene] = (effect, response)
    for position, symbol in enumerate(POSITIVE_CONTROLS):
        table[symbol] = (-1.4 + 0.6 * position, -1.1)
    for symbol in NEGATIVE_CONTROLS:
        table[symbol] = (-0.05, 0.1)
    return table
