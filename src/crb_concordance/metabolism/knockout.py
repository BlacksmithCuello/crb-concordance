"""Feasibility of alternative pathways under suppression of a candidate gene.

Ref: Sec. 4.2 item (3) (the flux-feasibility agent derives a context-specific model
and assesses the feasibility of alternative pathways suggested by suppression of g);
Sec. 4.4 (the flux-feasibility device is also assessed in isolation against the
dependency of the same cell lines).
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.metabolism import fba
from crb_concordance.metabolism.catalogue import DRAIN, EXCHANGE, TRANSPORT, pathway_reactions
from crb_concordance.metabolism.model import StoichiometricModel

SKIP_PATHWAYS: tuple[str, ...] = (EXCHANGE, DRAIN, "biomass")
CAPACITY_FRACTION = 0.999


@dataclass(frozen=True, slots=True)
class PathwayFeasibility:
    pathway: str
    baseline_capacity: float
    suppressed_capacity: float

    @property
    def retention(self) -> float:
        if self.baseline_capacity <= 1e-12:
            return 1.0
        return self.suppressed_capacity / self.baseline_capacity

    @property
    def loss(self) -> float:
        return 1.0 - self.retention

    def as_dict(self) -> dict[str, float | str]:
        return {
            "pathway": self.pathway,
            "baseline_capacity": self.baseline_capacity,
            "suppressed_capacity": self.suppressed_capacity,
            "retention": self.retention,
            "loss": self.loss,
        }


@dataclass(frozen=True, slots=True)
class SuppressionAssessment:
    """The flux-feasibility evidence for one candidate gene."""

    gene: str
    baseline_growth: float
    suppressed_growth: float
    blocked_reactions: tuple[str, ...]
    pathway_feasibility: tuple[PathwayFeasibility, ...]

    @property
    def growth_feasibility_loss(self) -> float:
        if self.baseline_growth <= 1e-12:
            return 0.0
        return 1.0 - self.suppressed_growth / self.baseline_growth

    @property
    def pathway_capacity_loss(self) -> float:
        material = [entry for entry in self.pathway_feasibility if entry.baseline_capacity > 1e-9]
        if not material:
            return 0.0
        return max(entry.loss for entry in material)

    @property
    def feasibility_score(self) -> float:
        """The graded flux evidence: growth feasibility and pathway capacity, equally weighted."""

        return 0.5 * self.growth_feasibility_loss + 0.5 * self.pathway_capacity_loss

    @property
    def gated(self) -> bool:
        return bool(self.blocked_reactions)

    def as_dict(self) -> dict[str, object]:
        return {
            "gene": self.gene,
            "gated": self.gated,
            "baseline_growth": self.baseline_growth,
            "suppressed_growth": self.suppressed_growth,
            "growth_feasibility_loss": self.growth_feasibility_loss,
            "pathway_capacity_loss": self.pathway_capacity_loss,
            "feasibility_score": self.feasibility_score,
            "blocked_reactions": list(self.blocked_reactions),
            "pathways": [entry.as_dict() for entry in self.pathway_feasibility],
        }


def pathway_capacity(model: StoichiometricModel, pathway: str, *, fraction: float = 1.0) -> float:
    """Total flux capacity of a pathway at a fixed fraction of the growth optimum."""

    entries = pathway_reactions(model.reactions, pathway)
    if not entries:
        return 0.0
    optimum = fba.max_biomass(model)
    if not optimum.optimal or optimum.objective_value <= 1e-12:
        return 0.0
    floor = float(fraction * optimum.objective_value)
    total = 0.0
    for entry in entries:
        total += abs(fba.maximise_reaction(model, entry.identifier, floor=floor))
    return total


def pathway_capacity_table(
    model: StoichiometricModel, *, fraction: float = CAPACITY_FRACTION
) -> dict[str, float]:
    pathways: list[str] = []
    for entry in model.reactions:
        name = entry.reaction.pathway
        if name in SKIP_PATHWAYS or name in pathways:
            continue
        pathways.append(name)
    return {
        name: pathway_capacity(model, name, fraction=fraction)
        for name in pathways
        if name != TRANSPORT
    }


def assess_gene_suppression(
    model: StoichiometricModel, gene: str, *, fraction: float = CAPACITY_FRACTION
) -> SuppressionAssessment:
    """Compare growth feasibility and pathway capacity with and without ``gene``."""

    baseline = fba.max_biomass(model)
    baseline_growth = float(baseline.objective_value) if baseline.optimal else 0.0
    suppressed_model = model.knock_out(gene)
    suppressed = fba.max_biomass(suppressed_model)
    suppressed_growth = float(suppressed.objective_value) if suppressed.optimal else 0.0
    baseline_capacities = pathway_capacity_table(model, fraction=fraction)
    if suppressed_growth <= 1e-12:
        feasibility = tuple(
            PathwayFeasibility(pathway=name, baseline_capacity=value, suppressed_capacity=0.0)
            for name, value in baseline_capacities.items()
        )
    else:
        suppressed_capacities = pathway_capacity_table(suppressed_model, fraction=fraction)
        feasibility = tuple(
            PathwayFeasibility(
                pathway=name,
                baseline_capacity=value,
                suppressed_capacity=suppressed_capacities.get(name, 0.0),
            )
            for name, value in baseline_capacities.items()
        )
    return SuppressionAssessment(
        gene=gene,
        baseline_growth=baseline_growth,
        suppressed_growth=suppressed_growth,
        blocked_reactions=model.blocked_reactions(gene),
        pathway_feasibility=feasibility,
    )


def essentiality_table(
    model: StoichiometricModel, genes: tuple[str, ...], *, threshold: float = 0.5
) -> dict[str, dict[str, float | bool]]:
    table: dict[str, dict[str, float | bool]] = {}
    for gene in genes:
        outcome = fba.knockout_growth(model, gene)
        table[gene] = {
            "growth_ratio": outcome.growth_ratio,
            "loss_of_fitness": outcome.loss_of_fitness,
            "essential": outcome.loss_of_fitness >= threshold,
        }
    return table


def dependency_rank(
    model: StoichiometricModel, genes: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    """Genes ordered by the flux-feasibility evidence of the paper's flux agent."""

    scored = [(gene, assess_gene_suppression(model, gene).feasibility_score) for gene in genes]
    return tuple(sorted(scored, key=lambda item: (-item[1], item[0])))


def synthetic_lethality(model: StoichiometricModel, first: str, second: str) -> float:
    """Extra growth loss from suppressing a second gene on top of the first."""

    single = fba.knockout_growth(model, first)
    combined_model = model.knock_out(first).knock_out(second)
    combined = fba.max_biomass(combined_model)
    if not combined.optimal or single.baseline <= 1e-12:
        return 1.0
    combined_ratio = float(combined.objective_value) / single.baseline
    return float(single.growth_ratio - combined_ratio)
