"""Metabolic layer: closed-form flux checks, gene rules and suppression scoring."""

from __future__ import annotations

import numpy as np
import pytest

from crb_concordance.metabolism.catalogue import (
    GLYCOLYSIS,
    NEGATIVE_CONTROLS,
    POSITIVE_CONTROLS,
    BoundedReaction,
    build_catalogue,
    gene_to_reactions,
    pathway_reactions,
    reaction_genes,
)
from crb_concordance.metabolism.fba import (
    flux_variability,
    knockout_growth,
    max_biomass,
    pathway_utilisation,
)
from crb_concordance.metabolism.knockout import (
    assess_gene_suppression,
    dependency_rank,
    pathway_capacity_table,
    synthetic_lethality,
)
from crb_concordance.metabolism.model import ModelError, StoichiometricModel
from crb_concordance.metabolism.reactions import (
    GprError,
    MetabolicReaction,
    gpr_genes,
    gpr_is_active,
    gpr_requires,
    parse_gpr,
)
from crb_concordance.metabolism.sampling import SamplingError, sample_fluxes


def _toy_entries() -> tuple[BoundedReaction, ...]:
    def entry(
        identifier: str,
        substrates: tuple[str, ...],
        products: tuple[str, ...],
        upper: float,
        gene: str,
    ) -> BoundedReaction:
        return BoundedReaction(
            reaction=MetabolicReaction(
                reaction_id=identifier,
                name=identifier,
                substrates=substrates,
                products=products,
                enzymes=(gene,),
                pathway="toy",
                gpr=gene,
            ),
            lower=0.0,
            upper=upper,
        )

    return (
        entry("FEED", (), ("sub",), 1.0, "ENV"),
        entry("DIMER", ("sub", "sub"), ("out",), 10.0, "G1"),
        entry("DRAIN", ("out",), (), 10.0, "ENV"),
    )


def test_fba_matches_the_hand_computed_optimum() -> None:
    model = StoichiometricModel.from_catalogue(_toy_entries(), name="toy").with_objective("DRAIN")
    result = max_biomass(model)
    assert result.optimal
    assert result.objective_value == pytest.approx(0.5)
    assert result.residual <= 1e-9


def test_fba_optimum_scales_with_the_feed_bound() -> None:
    model = StoichiometricModel.from_catalogue(_toy_entries(), name="toy").with_objective("DRAIN")
    halved = model.with_reaction_bounds("FEED", 0.0, 0.5)
    assert max_biomass(halved).objective_value == pytest.approx(0.25)


def test_dimer_gene_is_essential_in_the_toy_network() -> None:
    model = StoichiometricModel.from_catalogue(_toy_entries(), name="toy").with_objective("DRAIN")
    outcome = knockout_growth(model, "G1")
    assert outcome.growth_ratio == pytest.approx(0.0)
    assert outcome.blocked_reactions == ("DIMER",)


def test_catalogue_is_feasible_and_balances_mass(flux_model) -> None:
    result = max_biomass(flux_model)
    assert result.optimal
    assert result.objective_value > 0.0
    assert result.residual <= 1e-7
    assert flux_model.is_feasible(result.fluxes)


def test_catalogue_reaction_bounds_are_ordered() -> None:
    catalogue = build_catalogue()
    assert catalogue
    assert all(entry.lower <= entry.upper for entry in catalogue)
    assert all(entry.reaction.gpr for entry in catalogue)


def test_every_control_gene_gates_at_least_one_reaction(flux_model) -> None:
    for gene in POSITIVE_CONTROLS + NEGATIVE_CONTROLS:
        assert flux_model.blocked_reactions(gene), gene


def test_knockout_ratio_stays_in_the_unit_interval(flux_model) -> None:
    for gene in ("SLC2A1", "LDHA", "G6PD", "FASN"):
        outcome = knockout_growth(flux_model, gene)
        assert 0.0 <= outcome.growth_ratio <= 1.0 + 1e-9
        assert 0.0 <= outcome.loss_of_fitness <= 1.0 + 1e-9


def test_gene_rules_are_parsed_and_evaluated() -> None:
    tree = parse_gpr("(HK2 or GCK) and ATP5F1A")
    assert gpr_genes(tree) == frozenset({"HK2", "GCK", "ATP5F1A"})
    assert gpr_is_active(tree, {"HK2"})
    assert not gpr_is_active(tree, {"ATP5F1A"})
    assert not gpr_is_active(tree, {"HK2", "GCK"})
    assert gpr_requires(tree, "ATP5F1A")
    assert not gpr_requires(tree, "HK2")


def test_malformed_gene_rules_are_rejected() -> None:
    with pytest.raises(GprError):
        parse_gpr("(HK2 or GCK")
    with pytest.raises(GprError):
        parse_gpr("")


def test_gene_index_covers_every_catalogue_gene() -> None:
    catalogue = build_catalogue()
    index = gene_to_reactions(catalogue)
    assert set(index) == set(reaction_genes(catalogue))
    for identifiers in index.values():
        assert identifiers
        assert all(
            any(entry.identifier == identifier for entry in catalogue) for identifier in identifiers
        )


def test_context_specific_restriction_silences_unsupported_rules(flux_model) -> None:
    genes = reaction_genes(build_catalogue())
    expressed = {gene: 1.0 for gene in genes}
    unrestricted = flux_model.restrict_to_expressed(expressed, threshold=0.5)
    baseline = max_biomass(flux_model).objective_value
    assert max_biomass(unrestricted).objective_value == pytest.approx(baseline)

    transport_only = {
        gene: (
            1.0
            if gene in {"SLC2A1", "SLC2A3", "SLC25A4", "SLC1A5", "CD36", "SLC16A9", "SLC1A4"}
            else 0.0
        )
        for gene in genes
    }
    restricted = flux_model.restrict_to_expressed(transport_only, threshold=0.5)
    zeroed = int(
        np.count_nonzero(restricted.upper == 0.0) - np.count_nonzero(flux_model.upper == 0.0)
    )
    assert zeroed > 0
    assert max_biomass(restricted).objective_value <= baseline + 1e-9


def test_flux_variability_brackets_the_optimum(flux_model) -> None:
    variability = flux_variability(flux_model, fraction=1.0, reactions=("LDHA", "OXPHOS"))
    result = max_biomass(flux_model)
    for entry in variability.ranges:
        position = flux_model.index_of(entry.reaction_id)
        assert entry.minimum - 1e-6 <= entry.maximum
        assert entry.minimum - 1e-6 <= result.fluxes[position] <= entry.maximum + 1e-6


def test_suppression_assessment_reports_a_graded_score(flux_model) -> None:
    for gene in ("SLC2A1", "LDHA", "SLC16A9", "CES1"):
        assessment = assess_gene_suppression(flux_model, gene)
        assert assessment.gated
        assert 0.0 <= assessment.feasibility_score <= 1.0
        assert assessment.feasibility_score == pytest.approx(
            0.5 * assessment.growth_feasibility_loss + 0.5 * assessment.pathway_capacity_loss
        )


def test_dependency_rank_is_sorted_and_complete(flux_model) -> None:
    genes = ("SLC2A1", "LDHA", "SLC16A9", "CES1", "G6PD")
    ranked = dependency_rank(flux_model, genes)
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert {gene for gene, _ in ranked} == set(genes)


def test_pathway_capacity_table_covers_the_named_pathways(flux_model) -> None:
    table = pathway_capacity_table(flux_model)
    assert table
    assert all(value >= 0.0 for value in table.values())
    assert table.get(GLYCOLYSIS, 0.0) > 0.0


def test_pathway_utilisation_is_finite(flux_model) -> None:
    value = pathway_utilisation(flux_model, GLYCOLYSIS, fraction=0.9)
    assert np.isfinite(value)
    assert value >= 0.0


def test_synthetic_lethality_is_bounded(flux_model) -> None:
    value = synthetic_lethality(flux_model, "SLC2A1", "LDHA")
    assert -1e-9 <= value <= 1.0 + 1e-9


def test_pathway_reactions_partition_the_catalogue() -> None:
    catalogue = build_catalogue()
    collected: set[str] = set()
    for pathway in {entry.reaction.pathway for entry in catalogue}:
        identifiers = {entry.identifier for entry in pathway_reactions(catalogue, pathway)}
        assert not (identifiers & collected)
        collected |= identifiers
    assert collected == {entry.identifier for entry in catalogue}


def test_flux_sampling_draws_inside_the_polytope(flux_model) -> None:
    samples = sample_fluxes(flux_model, n_samples=40, seed=3, optimum_fraction=0.8)
    assert samples.samples.shape == (40, flux_model.n_reactions)
    for row in samples.samples:
        assert flux_model.is_feasible(row, tolerance=1e-5)
    assert samples.column("LDHA").shape == (40,)


def test_flux_sampling_rejects_an_impossible_request(flux_model) -> None:
    with pytest.raises(SamplingError):
        sample_fluxes(flux_model, n_samples=0, seed=1)


def test_model_rejects_an_unknown_objective(flux_model) -> None:
    with pytest.raises(ModelError):
        flux_model.with_objective("NOT_A_REACTION")


def test_unassigned_gene_returns_no_gated_reaction(flux_model) -> None:
    assert not flux_model.has_gene_rule("NOT_A_GENE")
    outcome = assess_gene_suppression(flux_model, "NOT_A_GENE")
    assert not outcome.gated
    assert outcome.feasibility_score == 0.0
