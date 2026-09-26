"""Context-specific stoichiometric model with gene-protein-reaction resolution.

Ref: Sec. 4.2 item (3) (a context-specific model is derived from the transcripts of
the tumour or cell line and the feasibility of alternative pathways is assessed
under suppression of g); Sec. 4.3 (reconstruction techniques).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from crb_concordance.metabolism.catalogue import (
    BIOMASS_DRAIN,
    BIOMASS_REACTION,
    ENVIRONMENT_MARKER,
    NON_GENE_MARKERS,
    BoundedReaction,
    build_catalogue,
)
from crb_concordance.metabolism.reactions import (
    GprNode,
    MetabolicReaction,
    gpr_is_active,
    gpr_requires,
    parse_gpr,
)


class ModelError(ValueError):
    """Raised when a model cannot be assembled or queried."""


@dataclass(slots=True)
class StoichiometricModel:
    """A stoichiometric matrix with flux bounds, gene rules and a growth objective."""

    metabolites: tuple[str, ...]
    reactions: tuple[BoundedReaction, ...]
    stoichiometry: sp.csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    objective_id: str = BIOMASS_DRAIN
    name: str = "reduced-human-gem"
    _index: dict[str, int] = field(default_factory=dict, repr=False)
    _gene_rules: dict[str, GprNode] = field(default_factory=dict, repr=False)
    _gene_blocked: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {entry.identifier: position for position, entry in enumerate(self.reactions)}
        self._gene_rules = {}
        blocked: dict[str, list[int]] = {}
        for position, entry in enumerate(self.reactions):
            tree = parse_gpr(entry.reaction.gpr)
            self._gene_rules[entry.identifier] = tree
            for symbol in entry.reaction.enzymes:
                if gpr_requires(tree, symbol):
                    blocked.setdefault(symbol, []).append(position)
        self._gene_blocked = {gene: tuple(sorted(rows)) for gene, rows in blocked.items()}

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    @property
    def n_metabolites(self) -> int:
        return len(self.metabolites)

    @property
    def genes(self) -> tuple[str, ...]:
        return tuple(sorted(self._gene_blocked))

    def index_of(self, reaction_id: str) -> int:
        try:
            return self._index[reaction_id]
        except KeyError as error:
            raise ModelError(f"unknown reaction: {reaction_id}") from error

    def objective_vector(self) -> np.ndarray:
        vector = np.zeros(self.n_reactions)
        vector[self.index_of(self.objective_id)] = 1.0
        return vector

    def flux_of(self, fluxes: np.ndarray, reaction_id: str) -> float:
        return float(fluxes[self.index_of(reaction_id)])

    def biomass_flux(self, fluxes: np.ndarray) -> float:
        return self.flux_of(fluxes, BIOMASS_REACTION)

    def knock_out(self, gene: str) -> StoichiometricModel:
        """Disable every reaction whose gene rule the loss of ``gene`` invalidates."""

        rows = self._gene_blocked.get(gene, ())
        if not rows:
            return self.copy()
        lower = self.lower.copy()
        upper = self.upper.copy()
        for row in rows:
            lower[row] = 0.0
            upper[row] = 0.0
        return self.with_bounds(lower, upper)

    def blocked_reactions(self, gene: str) -> tuple[str, ...]:
        return tuple(self.reactions[row].identifier for row in self._gene_blocked.get(gene, ()))

    def has_gene_rule(self, gene: str) -> bool:
        return bool(self._gene_blocked.get(gene))

    def with_bounds(self, lower: np.ndarray, upper: np.ndarray) -> StoichiometricModel:
        if lower.shape != (self.n_reactions,) or upper.shape != (self.n_reactions,):
            raise ModelError("bound vectors must match the reaction count")
        return StoichiometricModel(
            metabolites=self.metabolites,
            reactions=self.reactions,
            stoichiometry=self.stoichiometry,
            lower=lower,
            upper=upper,
            objective_id=self.objective_id,
            name=self.name,
        )

    def with_reaction_bounds(
        self, reaction_id: str, lower: float, upper: float
    ) -> StoichiometricModel:
        position = self.index_of(reaction_id)
        new_lower = self.lower.copy()
        new_upper = self.upper.copy()
        new_lower[position] = lower
        new_upper[position] = upper
        return self.with_bounds(new_lower, new_upper)

    def with_medium(self, caps: dict[str, float]) -> StoichiometricModel:
        """Rescale the declared medium caps, expressed as uptake upper bounds."""

        new_upper = self.upper.copy()
        new_lower = self.lower.copy()
        for reaction_id, cap in caps.items():
            position = self.index_of(reaction_id)
            new_upper[position] = float(cap)
            if new_lower[position] < 0.0:
                new_lower[position] = -float(cap)
        return self.with_bounds(new_lower, new_upper)

    def restrict_to_expressed(
        self,
        expression: dict[str, float],
        *,
        quantile: float = 0.25,
        threshold: float | None = None,
    ) -> StoichiometricModel:
        """Derive a context-specific model by silencing unsupported enzyme sets.

        A reaction is retained when at least one gene in its rule reaches the
        expression cutoff, which is the standard way to enforce a context without
        discarding complexes that are supported by a partner subunit.
        """

        values = np.asarray(list(expression.values()), dtype=float)
        if threshold is None:
            threshold = float(np.quantile(values, quantile)) if values.size else 0.0
        active = {gene for gene, value in expression.items() if value >= threshold}
        lower = self.lower.copy()
        upper = self.upper.copy()
        for position, entry in enumerate(self.reactions):
            inactive = {
                gene
                for gene in entry.reaction.enzymes
                if gene not in active and gene not in NON_GENE_MARKERS
            }
            if inactive and not gpr_is_active(self._gene_rules[entry.identifier], inactive):
                lower[position] = 0.0
                upper[position] = 0.0
        return self.with_bounds(lower, upper)

    def copy(self) -> StoichiometricModel:
        return self.with_bounds(self.lower.copy(), self.upper.copy())

    def bound_table(self) -> dict[str, tuple[float, float]]:
        return {
            entry.identifier: (float(self.lower[position]), float(self.upper[position]))
            for position, entry in enumerate(self.reactions)
        }

    def mass_balance_residual(self, fluxes: np.ndarray, *, tolerance: float = 1e-7) -> float:
        residual = self.stoichiometry.dot(np.asarray(fluxes, dtype=float))
        return float(np.max(np.abs(residual))) if residual.size else 0.0

    def is_feasible(self, fluxes: np.ndarray, *, tolerance: float = 1e-6) -> bool:
        flux = np.asarray(fluxes, dtype=float)
        if np.any(flux < self.lower - tolerance) or np.any(flux > self.upper + tolerance):
            return False
        return self.mass_balance_residual(flux, tolerance=tolerance) <= tolerance

    @classmethod
    def from_catalogue(
        cls,
        catalogue: tuple[BoundedReaction, ...] | None = None,
        *,
        name: str = "reduced-human-gem",
    ) -> StoichiometricModel:
        entries = catalogue if catalogue is not None else build_catalogue()
        metabolites: list[str] = []
        for entry in entries:
            for species in entry.reaction.substrates + entry.reaction.products:
                if species not in metabolites:
                    metabolites.append(species)
        column = {species: position for position, species in enumerate(metabolites)}
        columns: list[int] = []
        rows: list[int] = []
        values: list[float] = []
        for reaction_position, entry in enumerate(entries):
            tally: Counter[str] = Counter()
            for species in entry.reaction.products:
                tally[species] += 1.0
            for species in entry.reaction.substrates:
                tally[species] -= 1.0
            for species, coefficient in tally.items():
                if coefficient == 0.0:
                    continue
                rows.append(column[species])
                columns.append(reaction_position)
                values.append(coefficient)
        matrix = sp.csr_matrix((values, (rows, columns)), shape=(len(metabolites), len(entries)))
        return cls(
            metabolites=tuple(metabolites),
            reactions=tuple(entries),
            stoichiometry=matrix,
            lower=np.asarray([entry.lower for entry in entries], dtype=float),
            upper=np.asarray([entry.upper for entry in entries], dtype=float),
            objective_id=BIOMASS_DRAIN,
            name=name,
        )

    @classmethod
    def from_human_gem(
        cls, path: str | Path, *, source: str = "Human-GEM v2.0.1"
    ) -> StoichiometricModel:
        """Read a released Human-GEM JSON export with the standard field names."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        reactions: list[BoundedReaction] = []
        for record in payload["reactions"]:
            participants = record.get("metabolites", {})
            substrates = tuple(
                name for name, coefficient in participants.items() if float(coefficient) < 0
            )
            products = tuple(
                name for name, coefficient in participants.items() if float(coefficient) > 0
            )
            rule = str(record.get("gene_reaction_rule", "") or record.get("id", ""))
            enzymes = tuple(
                sorted(
                    {
                        token
                        for token in rule.replace("(", " ").replace(")", " ").split()
                        if token.lower() not in {"and", "or"}
                    }
                )
            ) or (ENVIRONMENT_MARKER,)
            entry = BoundedReaction(
                reaction=MetabolicReaction(
                    reaction_id=str(record["id"]),
                    name=str(record.get("name", record["id"])),
                    substrates=substrates,
                    products=products,
                    enzymes=enzymes,
                    pathway=str(record.get("subsystem", "unassigned") or "unassigned"),
                    reversible=float(record.get("lower_bound", 0.0)) < 0.0,
                    gpr=rule or " or ".join(enzymes),
                ),
                lower=float(record.get("lower_bound", 0.0)),
                upper=float(record.get("upper_bound", 1000.0)),
            )
            entry.reaction.validate()
            reactions.append(entry)
        if not reactions:
            raise ModelError(f"no reactions found in {path}")
        model = cls.from_catalogue(tuple(reactions), name=source)
        objective = next(
            (entry for entry in reactions if entry.identifier == BIOMASS_REACTION), reactions[-1]
        )
        return model.with_objective(objective.identifier)

    def with_objective(self, reaction_id: str) -> StoichiometricModel:
        if reaction_id not in self._index:
            raise ModelError(f"objective reaction absent from the model: {reaction_id}")
        return StoichiometricModel(
            metabolites=self.metabolites,
            reactions=self.reactions,
            stoichiometry=self.stoichiometry,
            lower=self.lower,
            upper=self.upper,
            objective_id=reaction_id,
            name=self.name,
        )
