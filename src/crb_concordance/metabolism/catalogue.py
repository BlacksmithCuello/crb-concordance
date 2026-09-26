"""Curated reaction catalogue covering the pathways named in the paper.

Ref: Sec. 4.3 (a context-specific model is extracted from Human-GEM v2.0.1 with
enzyme constraints, complexity reduction and missing-reaction completion);
Sec. 1 (the lactate/glycolysis, fatty-acid-oxidation-carnitine, oxidative
phosphorylation, ferroptosis/cholesterol and KRAS nucleotide-repair axes).

Stoichiometry is unit-coefficient, so a repeated species appears once per side.
The catalogue is a reduced reconstruction of the named pathways rather than the
full genome-scale model. Cytosolic and mitochondrial redox are held as separate
pools with a rate-limited shuttle between them, because a single shared NAD/NADH
pool would let oxidative phosphorylation clear glycolytic reducing equivalents
directly and would remove the lactate axis the paper is about. Exchange caps are
the declared default medium; ``StoichiometricModel.with_medium`` overrides them and
``StoichiometricModel.from_human_gem`` reads a released Human-GEM JSON export.
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.metabolism.reactions import MetabolicReaction

BIOMASS_REACTION = "MAR09034_biomass"
BIOMASS_DRAIN = "DM_biomass"
INTERNAL_CAP = 1000.0
BIOMASS_MARKER = "BIOMASS_MARKER"
ENVIRONMENT_MARKER = "ENVIRONMENT"
NON_GENE_MARKERS: frozenset[str] = frozenset({BIOMASS_MARKER, ENVIRONMENT_MARKER})

GLYCOLYSIS = "glycolysis-lactate"
REDOX_SHUTTLE = "redox-shuttle"
OXPHOS = "oxidative-phosphorylation"
FAO_CARNITINE = "fatty-acid-oxidation-carnitine-shuttle"
FATTY_ACID_SYNTHESIS = "fatty-acid-synthesis"
FERROPTOSIS = "ferroptosis-cholesterol"
GLUTAMINOLYSIS = "glutaminolysis"
PENTOSE_PHOSPHATE = "pentose-phosphate"
ONE_CARBON = "one-carbon"
NUCLEOTIDE_REPAIR = "nucleotide-dna-repair"
TCA = "tca-cycle"
TRANSPORT = "transport"
EXCHANGE = "exchange"
DRAIN = "drain"

POSITIVE_CONTROLS: tuple[str, ...] = ("SLC2A1", "SLC16A1")
NEGATIVE_CONTROLS: tuple[str, ...] = ("SLC16A9", "CPT1A", "CES1")
CONTROL_ALIASES: dict[str, str] = {
    "GLUT1": "SLC2A1",
    "MCT1": "SLC16A1",
    "MCT9": "SLC16A9",
}

UNCAPPED = INTERNAL_CAP
SHUTTLE_CAP = 1.0


@dataclass(frozen=True, slots=True)
class BoundedReaction:
    reaction: MetabolicReaction
    lower: float
    upper: float

    @property
    def identifier(self) -> str:
        return self.reaction.reaction_id


def _reaction(
    reaction_id: str,
    name: str,
    substrates: tuple[str, ...],
    products: tuple[str, ...],
    enzymes: tuple[str, ...],
    pathway: str,
    *,
    lower: float = 0.0,
    upper: float = INTERNAL_CAP,
    gpr: str = "",
    reversible: bool = False,
) -> BoundedReaction:
    record = MetabolicReaction(
        reaction_id=reaction_id,
        name=name,
        substrates=substrates,
        products=products,
        enzymes=enzymes,
        pathway=pathway,
        reversible=reversible,
        gpr=gpr or " or ".join(enzymes),
    )
    record.validate()
    return BoundedReaction(reaction=record, lower=lower, upper=upper)


def _exchange(
    reaction_id: str,
    name: str,
    substrates: tuple[str, ...],
    products: tuple[str, ...],
    *,
    upper: float = INTERNAL_CAP,
) -> BoundedReaction:
    return _reaction(
        reaction_id, name, substrates, products, (ENVIRONMENT_MARKER,), EXCHANGE, upper=upper
    )


def build_catalogue() -> tuple[BoundedReaction, ...]:
    """The reduced reconstruction, ordered so that metabolite indices stay stable."""

    return (
        _exchange("EX_glc_D_e", "D-glucose pool", (), ("glc_D_e",), upper=10.0),
        _exchange("EX_o2_e", "oxygen pool", (), ("o2_e",), upper=6.0),
        _exchange("EX_gln_e", "glutamine pool", (), ("gln_e",), upper=2.0),
        _exchange("EX_fa_e", "free fatty acid pool", (), ("fa_e",), upper=2.0),
        _exchange("EX_carn_e", "carnitine pool", (), ("carn_e",), upper=5.0),
        _exchange("EX_ser_e", "serine pool", (), ("ser_e",), upper=1.0),
        _exchange("EX_lac_e", "lactate sink", ("lac_e",), ()),
        _exchange("EX_co2_e", "carbon dioxide sink", ("co2_c",), ()),
        _exchange("EX_h2o_e", "water sink", ("h2o_c",), ()),
        _exchange("EX_gly_e", "glycine sink", ("gly_c",), ()),
        _reaction("DM_biomass", "biomass drain", ("biomass_c",), (), (BIOMASS_MARKER,), DRAIN),
        _reaction(
            "GLUT1_transport",
            "glucose transport across the plasma membrane",
            ("glc_D_e",),
            ("glc_D_c",),
            ("SLC2A1",),
            TRANSPORT,
            upper=10.0,
            gpr="SLC2A1",
        ),
        _reaction(
            "GLUT3_transport",
            "glucose transport via the GLUT3 isoform",
            ("glc_D_e",),
            ("glc_D_c",),
            ("SLC2A3",),
            TRANSPORT,
            upper=0.2,
            gpr="SLC2A3",
        ),
        _reaction(
            "O2_transport",
            "oxygen transport",
            ("o2_e",),
            ("o2_c",),
            ("SLC25A4",),
            TRANSPORT,
            upper=6.0,
        ),
        _reaction(
            "GLN_transport",
            "glutamine transport",
            ("gln_e",),
            ("gln_c",),
            ("SLC1A5",),
            TRANSPORT,
            upper=2.0,
        ),
        _reaction(
            "FA_transport",
            "fatty acid transport",
            ("fa_e",),
            ("fa_c",),
            ("CD36",),
            TRANSPORT,
            upper=2.0,
        ),
        _reaction(
            "CARN_transport",
            "carnitine transport",
            ("carn_e",),
            ("carn_c",),
            ("SLC16A9",),
            TRANSPORT,
            upper=5.0,
        ),
        _reaction(
            "SER_transport",
            "serine transport",
            ("ser_e",),
            ("ser_c",),
            ("SLC1A4",),
            TRANSPORT,
            upper=1.0,
        ),
        _reaction(
            "PHGDH",
            "phosphoglycerate dehydrogenase",
            ("p3g_c", "nad_c"),
            ("php_c", "nadh_c"),
            ("PHGDH",),
            ONE_CARBON,
        ),
        _reaction(
            "PSAT1",
            "phosphoserine aminotransferase",
            ("php_c", "glu_c"),
            ("pser_c", "akg_c"),
            ("PSAT1",),
            ONE_CARBON,
        ),
        _reaction(
            "PSPH", "phosphoserine phosphatase", ("pser_c",), ("ser_c",), ("PSPH",), ONE_CARBON
        ),
        _reaction(
            "HEX",
            "hexokinase",
            ("glc_D_c", "atp_c"),
            ("g6p_c", "adp_c"),
            ("HK2", "GCK"),
            GLYCOLYSIS,
            gpr="HK2 or GCK",
        ),
        _reaction(
            "PGI",
            "phosphoglucose isomerase",
            ("g6p_c",),
            ("f6p_c",),
            ("GPI",),
            GLYCOLYSIS,
            lower=-UNCAPPED,
            reversible=True,
        ),
        _reaction(
            "PFK",
            "phosphofructokinase",
            ("f6p_c", "atp_c"),
            ("fbp_c", "adp_c"),
            ("PFKM", "PFKP"),
            GLYCOLYSIS,
            gpr="PFKM or PFKP",
        ),
        _reaction(
            "ALDO",
            "aldolase",
            ("fbp_c",),
            ("dhap_c", "gap_c"),
            ("ALDOA",),
            GLYCOLYSIS,
            lower=-UNCAPPED,
            reversible=True,
        ),
        _reaction(
            "TPI",
            "triose phosphate isomerase",
            ("dhap_c",),
            ("gap_c",),
            ("TPI1",),
            GLYCOLYSIS,
            lower=-UNCAPPED,
            reversible=True,
        ),
        _reaction(
            "GAPDH",
            "glyceraldehyde phosphate dehydrogenase",
            ("gap_c", "nad_c"),
            ("dpg_c", "nadh_c"),
            ("GAPDH",),
            GLYCOLYSIS,
        ),
        _reaction(
            "PGK",
            "phosphoglycerate kinase",
            ("dpg_c", "adp_c"),
            ("p3g_c", "atp_c"),
            ("PGK1",),
            GLYCOLYSIS,
        ),
        _reaction(
            "PGAM",
            "phosphoglycerate mutase",
            ("p3g_c",),
            ("p2g_c",),
            ("PGAM1",),
            GLYCOLYSIS,
            lower=-UNCAPPED,
            reversible=True,
        ),
        _reaction("ENO", "enolase", ("p2g_c",), ("pep_c",), ("ENO1",), GLYCOLYSIS),
        _reaction(
            "PKM", "pyruvate kinase", ("pep_c", "adp_c"), ("pyr_c", "atp_c"), ("PKM",), GLYCOLYSIS
        ),
        _reaction(
            "LDHA",
            "lactate dehydrogenase A",
            ("pyr_c", "nadh_c"),
            ("lac_c", "nad_c"),
            ("LDHA",),
            GLYCOLYSIS,
            lower=-UNCAPPED,
            reversible=True,
        ),
        _reaction(
            "MCT1_export",
            "monocarboxylate export of lactate",
            ("lac_c",),
            ("lac_e",),
            ("SLC16A1",),
            GLYCOLYSIS,
            gpr="SLC16A1",
        ),
        _reaction(
            "MCT4_export",
            "monocarboxylate export via MCT4",
            ("lac_c",),
            ("lac_e",),
            ("SLC16A3",),
            GLYCOLYSIS,
            upper=2.0,
            gpr="SLC16A3",
        ),
        _reaction(
            "G3P_shuttle",
            "glycerol-3-phosphate redox shuttle",
            ("nadh_c", "nad_m"),
            ("nad_c", "nadh_m"),
            ("GPD2",),
            REDOX_SHUTTLE,
            upper=SHUTTLE_CAP,
            gpr="GPD2",
        ),
        _reaction(
            "PDH",
            "pyruvate dehydrogenase",
            ("pyr_c", "nad_m", "coa_c"),
            ("accoa_c", "nadh_m", "co2_c"),
            ("PDHA1",),
            TCA,
        ),
        _reaction("CS", "citrate synthase", ("accoa_c", "oaa_c"), ("cit_c", "coa_c"), ("CS",), TCA),
        _reaction(
            "ACO",
            "aconitase",
            ("cit_c",),
            ("icit_c",),
            ("ACO2",),
            TCA,
            lower=-UNCAPPED,
            reversible=True,
        ),
        _reaction(
            "IDH",
            "isocitrate dehydrogenase",
            ("icit_c", "nad_m"),
            ("akg_c", "nadh_m", "co2_c"),
            ("IDH3A", "IDH2"),
            TCA,
            gpr="IDH3A or IDH2",
        ),
        _reaction(
            "OGDH",
            "oxoglutarate dehydrogenase",
            ("akg_c", "nad_m", "coa_c"),
            ("succoa_c", "nadh_m", "co2_c"),
            ("OGDH",),
            TCA,
        ),
        _reaction(
            "SUCLA",
            "succinyl-CoA ligase",
            ("succoa_c", "adp_c"),
            ("succ_c", "atp_c", "coa_c"),
            ("SUCLA2",),
            TCA,
        ),
        _reaction("SDH", "succinate dehydrogenase", ("succ_c",), ("mal_c",), ("SDHA",), TCA),
        _reaction(
            "MDH", "malate dehydrogenase", ("mal_c", "nad_m"), ("oaa_c", "nadh_m"), ("MDH2",), TCA
        ),
        _reaction(
            "OXPHOS",
            "mitochondrial oxidative phosphorylation",
            (
                "nadh_m",
                "nadh_m",
                "o2_c",
            )
            + ("adp_c",) * 8,
            ("nad_m", "nad_m", "h2o_c", "h2o_c") + ("atp_c",) * 8,
            ("NDUFS1", "COX5A", "ATP5F1A"),
            OXPHOS,
            gpr="NDUFS1 and COX5A and ATP5F1A",
        ),
        _reaction(
            "CPT1A_shuttle",
            "carnitine palmitoyltransferase loading",
            ("carn_c", "fa_c", "coa_c"),
            ("acylcarn_c",),
            ("CPT1A",),
            FAO_CARNITINE,
            gpr="CPT1A",
        ),
        _reaction(
            "FAO_beta",
            "mitochondrial beta-oxidation",
            ("acylcarn_c", "coa_c", "nad_m", "nad_m"),
            ("accoa_c", "carn_c", "nadh_m", "nadh_m"),
            ("ACADM", "HADHA"),
            FAO_CARNITINE,
            gpr="ACADM or HADHA",
        ),
        _reaction(
            "ACC",
            "acetyl-CoA carboxylase",
            ("accoa_c", "atp_c", "co2_c"),
            ("malonyl_c", "adp_c"),
            ("ACACA",),
            FATTY_ACID_SYNTHESIS,
            gpr="ACACA",
        ),
        _reaction(
            "FASN",
            "fatty acid synthase",
            ("malonyl_c", "nadph_c", "nadph_c"),
            ("palm_c", "co2_c", "coa_c", "nadp_c", "nadp_c"),
            ("FASN",),
            FATTY_ACID_SYNTHESIS,
            gpr="FASN",
        ),
        _reaction(
            "SCD",
            "stearoyl-CoA desaturase",
            ("palm_c", "o2_c", "nadph_c"),
            ("palm_unsat_c", "nadp_c", "h2o_c"),
            ("SCD",),
            FERROPTOSIS,
            gpr="SCD",
        ),
        _reaction(
            "ACSL4",
            "acyl-CoA synthetase long chain family member 4",
            ("palm_unsat_c", "coa_c", "atp_c"),
            ("palm_unsat_coa_c", "adp_c"),
            ("ACSL4",),
            FERROPTOSIS,
            gpr="ACSL4",
        ),
        _reaction(
            "LPOX",
            "lipid peroxidation of the unsaturated acyl chain",
            ("palm_unsat_coa_c", "o2_c"),
            ("lpoa_c", "h2o_c"),
            ("ALOX15",),
            FERROPTOSIS,
            gpr="ALOX15",
        ),
        _reaction(
            "GPX4",
            "glutathione peroxidase 4",
            ("lpoa_c", "nadph_c"),
            ("palm_unsat_coa_c", "nadp_c"),
            ("GPX4",),
            FERROPTOSIS,
            gpr="GPX4",
        ),
        _reaction(
            "CES1_detox",
            "carboxylesterase 1 de-esterification",
            ("lpoa_c",),
            ("palm_unsat_c", "coa_c"),
            ("CES1",),
            FERROPTOSIS,
            gpr="CES1",
        ),
        _reaction(
            "HMGCR",
            "HMG-CoA reductase",
            ("accoa_c", "nadph_c"),
            ("chol_c", "coa_c", "nadp_c"),
            ("HMGCR",),
            FERROPTOSIS,
            gpr="HMGCR",
        ),
        _reaction("GLS", "glutaminase", ("gln_c",), ("glu_c",), ("GLS",), GLUTAMINOLYSIS),
        _reaction(
            "GLUD1",
            "glutamate dehydrogenase",
            ("glu_c", "nad_m"),
            ("akg_c", "nadh_m"),
            ("GLUD1",),
            GLUTAMINOLYSIS,
        ),
        _reaction(
            "G6PD",
            "glucose-6-phosphate dehydrogenase",
            ("g6p_c", "nadp_c"),
            ("pgl_c", "nadph_c"),
            ("G6PD",),
            PENTOSE_PHOSPHATE,
        ),
        _reaction("TKT", "transketolase", ("pgl_c",), ("r5p_c",), ("TKT",), PENTOSE_PHOSPHATE),
        _reaction(
            "ME1",
            "malic enzyme",
            ("mal_c", "nadp_c"),
            ("pyr_c", "nadph_c", "co2_c"),
            ("ME1",),
            PENTOSE_PHOSPHATE,
        ),
        _reaction(
            "SHMT2",
            "serine hydroxymethyltransferase",
            ("ser_c",),
            ("gly_c", "thf_c"),
            ("SHMT2",),
            ONE_CARBON,
        ),
        _reaction(
            "MTHFD2",
            "methylenetetrahydrofolate dehydrogenase",
            ("thf_c", "nadp_c"),
            ("mthf_c", "nadph_c"),
            ("MTHFD2",),
            ONE_CARBON,
        ),
        _reaction(
            "DHFR",
            "dihydrofolate reductase",
            ("mthf_c", "nadph_c"),
            ("thf_c", "nadp_c"),
            ("DHFR",),
            ONE_CARBON,
        ),
        _reaction(
            "TK1",
            "thymidine kinase",
            ("mthf_c", "adp_c"),
            ("dtmp_c", "atp_c"),
            ("TK1",),
            NUCLEOTIDE_REPAIR,
        ),
        _reaction(
            "RRM2",
            "ribonucleotide reductase",
            ("atp_c", "nadph_c"),
            ("dndp_c", "adp_c", "nadp_c"),
            ("RRM2",),
            NUCLEOTIDE_REPAIR,
        ),
        _reaction(
            "NEK8_repair",
            "NEK8-dependent DNA-repair signalling",
            ("dndp_c", "atp_c"),
            ("repair_c", "adp_c"),
            ("NEK8",),
            NUCLEOTIDE_REPAIR,
        ),
        _reaction(
            BIOMASS_REACTION,
            "biomass precursor drain",
            ("atp_c", "palm_c", "r5p_c", "dtmp_c", "oaa_c", "chol_c", "repair_c"),
            ("adp_c", "biomass_c"),
            ("BIOMASS_MARKER",),
            "biomass",
            gpr="BIOMASS_MARKER",
        ),
    )


def catalogue_index(reactions: tuple[BoundedReaction, ...]) -> dict[str, BoundedReaction]:
    return {entry.identifier: entry for entry in reactions}


def pathway_reactions(
    reactions: tuple[BoundedReaction, ...], pathway: str
) -> tuple[BoundedReaction, ...]:
    return tuple(entry for entry in reactions if entry.reaction.pathway == pathway)


def gene_to_reactions(reactions: tuple[BoundedReaction, ...]) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for entry in reactions:
        for gene in entry.reaction.enzymes:
            if gene in NON_GENE_MARKERS:
                continue
            identifiers = mapping.setdefault(gene, [])
            if entry.identifier not in identifiers:
                identifiers.append(entry.identifier)
    return {gene: tuple(sorted(ids)) for gene, ids in mapping.items()}


def reaction_genes(reactions: tuple[BoundedReaction, ...]) -> tuple[str, ...]:
    genes: set[str] = set()
    for entry in reactions:
        genes.update(gene for gene in entry.reaction.enzymes if gene not in NON_GENE_MARKERS)
    return tuple(sorted(genes))


def resolve_gene_symbol(symbol: str) -> str:
    return CONTROL_ALIASES.get(symbol.upper(), symbol.upper())


def alternative_uptake_genes(reactions: tuple[BoundedReaction, ...], gene: str) -> tuple[str, ...]:
    """Genes whose reactions can carry carbon when ``gene`` is suppressed."""

    blocked = set(gene_to_reactions(reactions).get(gene, ()))
    if not blocked:
        return ()
    alternatives: set[str] = set()
    for entry in reactions:
        if entry.reaction.pathway in (TRANSPORT, EXCHANGE, DRAIN):
            continue
        if entry.identifier in blocked:
            continue
        alternatives.update(
            candidate for candidate in entry.reaction.enzymes if candidate not in NON_GENE_MARKERS
        )
    return tuple(sorted(alternatives))
