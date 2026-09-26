"""Shared run-time assembly for the command-line entry points.

Ref: Sec. 4.2 (the five-agent architecture is assembled once per run); Sec. 4.3 (the
substrate, the flux model and the cohort layer are the four evidence sources).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from crb_concordance.agents.base import AgentContext
from crb_concordance.agents.mapping import MassMapper, MassMappingParams
from crb_concordance.belief.discount import DiscountRates
from crb_concordance.calibration.discount_fit import CalibrationReport, calibrate
from crb_concordance.cohorts.control_panel import ControlPanel, build_control_panel
from crb_concordance.cohorts.generator import (
    CohortDesign,
    generate_cohort,
    generate_expression,
)
from crb_concordance.cohorts.schema import ClinicalRecord
from crb_concordance.graph.generator import (
    SubstrateBundle,
    SubstrateSpec,
    build_substrate,
    dependency_table_from_substrate,
)
from crb_concordance.graph.paths import PathFinder
from crb_concordance.graph.substrate import NodeKind, Relation
from crb_concordance.metabolism.model import StoichiometricModel
from crb_concordance.utils.config import ExperimentConfig, load_experiment
from crb_concordance.utils.types import MODALITY_ORDER, Modality

DEFAULT_CONFIG_ROOT = "configs"

PATH_RELATIONS: tuple[Relation, ...] = (
    Relation.GENE_PATHWAY_MEMBERSHIP,
    Relation.PATHWAY_DISEASE_ASSOCIATION,
    Relation.GENE_PHENOTYPE_ASSOCIATION,
    Relation.REACTION_ENZYME,
    Relation.METABOLITE_PATHWAY,
    Relation.METABOLIC_REACTION,
    Relation.PROTEIN_INTERACTION,
    Relation.GENE_DISEASE_ASSOCIATION,
    Relation.DRUG_TARGET,
)

PATH_KINDS: tuple[NodeKind, ...] = (
    NodeKind.PATHWAY,
    NodeKind.PHENOTYPE,
    NodeKind.GENE,
    NodeKind.REACTION,
    NodeKind.METABOLITE,
    NodeKind.DISEASE,
)


class RuntimeError_(ValueError):
    """Raised when the run-time cannot be assembled from the supplied settings."""


@dataclass(slots=True)
class StudyRuntime:
    """Everything a study needs, assembled once."""

    substrate: SubstrateBundle
    finder: PathFinder
    model: StoichiometricModel
    records: tuple[ClinicalRecord, ...]
    expression: dict[str, object]
    panel: ControlPanel
    calibration: CalibrationReport
    mapper: MassMapper
    context: AgentContext
    config: ExperimentConfig
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def rates(self) -> DiscountRates:
        return self.calibration.pooled

    def candidate_pool(self) -> tuple[str, ...]:
        return self.substrate.candidate_pool


def build_mapper(params: dict[str, MassMappingParams] | None) -> MassMapper:
    return MassMapper(params)


def _modality_params(config: ExperimentConfig) -> dict[Modality, MassMappingParams] | None:
    block = config.get("model.mass_map")
    if not isinstance(block, dict):
        return None
    params: dict[Modality, MassMappingParams] = {}
    for modality in MODALITY_ORDER:
        entry = block.get(modality.value)
        if not isinstance(entry, dict):
            return None
        params[modality] = MassMappingParams(
            slope=float(entry.get("slope", 6.0)),
            offset=float(entry.get("offset", -3.0)),
            commitment=float(entry.get("commitment", 0.9)),
        )
    return params


def build_runtime(
    *,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    experiment: str = "main",
    overrides: list[str] | None = None,
    cohort_size: int | None = None,
    n_candidates: int | None = None,
) -> StudyRuntime:
    """Assemble the substrate, flux model, cohorts, panel, calibration and context."""

    config = load_experiment(config_root, experiment, overrides=overrides)
    spec = SubstrateSpec(
        n_background_genes=int(config.get("data.n_background_genes", 72)),
        n_drugs=int(config.get("data.n_drugs", 24)),
        seed=int(config.get("data.substrate_seed", 20260101)),
        curation_cutoff=str(config.get("data.curation_cutoff", "2026-01-01")),
    )
    substrate = build_substrate(spec)
    finder = PathFinder(
        substrate.graph,
        max_hops=int(config.get("model.max_hops", 3)),
        allowed_relations=PATH_RELATIONS,
        allowed_kinds=PATH_KINDS,
    )
    model = StoichiometricModel.from_catalogue().with_medium(_medium(config))
    level = cohort_size or int(config.get("data.cohort_size", 0))
    design = (
        CohortDesign()
        if not level
        else CohortDesign(
            retrospective_site_counts=(level // 3, level // 3, level - 2 * (level // 3)),
            prospective_size=int(config.get("data.prospective_size", 825)),
        )
    )
    if design.retrospective_size < 3000:
        design = CohortDesign(prospective_size=design.prospective_size)
    records = generate_cohort(design)
    genes = substrate.candidate_pool
    n_genes = n_candidates or int(config.get("data.expression_genes", 48))
    expression = generate_expression(records, tuple(genes[:n_genes]))
    panel = build_control_panel(
        dispersion=float(config.get("model.panel_dispersion", 0.16)),
        seed=int(config.get("model.panel_seed", 20260101)),
    )
    calibration = calibrate(panel)
    mapper = build_mapper(_modality_params(config))
    context = AgentContext(
        finder=finder,
        phenotype=substrate.phenotype,
        model=model,
        dependency_table=dependency_table_from_substrate(substrate),
        profiles=expression,  # type: ignore[arg-type]
        records=records,
        panel=panel,
    )
    return StudyRuntime(
        substrate=substrate,
        finder=finder,
        model=model,
        records=records,
        expression=expression,
        panel=panel,
        calibration=calibration,
        mapper=mapper,
        context=context,
        config=config,
        notes={
            "cohort_design": design.as_dict(),
            "substrate": substrate.as_dict(),
            "calibration_rates": calibration.pooled.as_dict(),
            "ignorance_floor": calibration.pooled.ignorance_floor(),
        },
    )


def _medium(config: ExperimentConfig) -> dict[str, float]:
    block = config.get("data.medium")
    if not isinstance(block, dict):
        return {}
    return {str(key): float(value) for key, value in block.items()}


def output_directory(experiment: str, *, root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else default_output_root()
    return base / experiment


def default_output_root() -> Path:
    return Path("runs")
