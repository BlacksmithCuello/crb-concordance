"""Reaction records and gene-protein-reaction logic.

Ref: Sec. 4.2 item (3) (a context-specific model is derived and the feasibility of
alternative pathways assessed under suppression of g); Sec. 4.3 (enzyme-constrained
and complexity-reduced reconstruction of Human-GEM v2.0.1).
"""

from __future__ import annotations

from dataclasses import dataclass


class GprError(ValueError):
    """Raised when a gene-protein-reaction expression cannot be parsed."""


class ReactionError(ValueError):
    """Raised when a reaction record is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class MetabolicReaction:
    """One stoichiometric reaction with its enzyme set and pathway assignment."""

    reaction_id: str
    name: str
    substrates: tuple[str, ...]
    products: tuple[str, ...]
    enzymes: tuple[str, ...]
    pathway: str
    reversible: bool = False
    gpr: str = ""

    def validate(self) -> None:
        if not self.reaction_id:
            raise ReactionError("reaction id must be non-empty")
        if not self.substrates and not self.products:
            raise ReactionError(f"reaction {self.reaction_id} has no participants")
        if not self.enzymes:
            raise ReactionError(f"reaction {self.reaction_id} has no enzyme assignment")
        if self.gpr:
            parse_gpr(self.gpr)

    @property
    def direction(self) -> str:
        return "reversible" if self.reversible else "forward"


@dataclass(frozen=True, slots=True)
class GprGene:
    symbol: str


@dataclass(frozen=True, slots=True)
class GprJunction:
    operator: str
    children: tuple[GprNode, ...]


GprNode = GprGene | GprJunction

_TOKEN_CLOSERS = {")"}


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    buffer = ""
    for character in text:
        if character in "()":
            if buffer:
                tokens.append(buffer)
                buffer = ""
            tokens.append(character)
        elif character.isspace():
            if buffer:
                tokens.append(buffer)
                buffer = ""
        else:
            buffer += character
    if buffer:
        tokens.append(buffer)
    return tokens


def parse_gpr(text: str) -> GprNode:
    """Parse an ``and``/``or`` gene-protein-reaction expression."""

    tokens = _tokenize(text)
    if not tokens:
        raise GprError("empty gene-protein-reaction expression")
    position = 0

    def parse_or() -> GprNode:
        nonlocal position
        node = parse_and()
        children = [node]
        while position < len(tokens) and tokens[position].lower() == "or":
            position += 1
            children.append(parse_and())
        if len(children) == 1:
            return children[0]
        return GprJunction(operator="or", children=tuple(children))

    def parse_and() -> GprNode:
        nonlocal position
        node = parse_atom()
        children = [node]
        while position < len(tokens) and tokens[position].lower() == "and":
            position += 1
            children.append(parse_atom())
        if len(children) == 1:
            return children[0]
        return GprJunction(operator="and", children=tuple(children))

    def parse_atom() -> GprNode:
        nonlocal position
        if position >= len(tokens):
            raise GprError("unexpected end of gene-protein-reaction expression")
        token = tokens[position]
        if token == "(":
            position += 1
            node = parse_or()
            if position >= len(tokens) or tokens[position] not in _TOKEN_CLOSERS:
                raise GprError("unbalanced parentheses in gene-protein-reaction expression")
            position += 1
            return node
        position += 1
        return GprGene(symbol=token)

    root = parse_or()
    if position != len(tokens):
        raise GprError(f"trailing tokens in gene-protein-reaction expression: {tokens[position:]}")
    return root


def gpr_genes(node: GprNode) -> frozenset[str]:
    if isinstance(node, GprGene):
        return frozenset({node.symbol})
    collected: set[str] = set()
    for child in node.children:
        collected |= gpr_genes(child)
    return frozenset(collected)


def gpr_is_active(node: GprNode, inactive: frozenset[str] | set[str]) -> bool:
    """Whether an enzyme complex can still form once ``inactive`` genes are lost."""

    if isinstance(node, GprGene):
        return node.symbol not in inactive
    if node.operator == "and":
        return all(gpr_is_active(child, inactive) for child in node.children)
    return any(gpr_is_active(child, inactive) for child in node.children)


def gpr_requires(node: GprNode, gene: str) -> bool:
    """Whether losing ``gene`` alone is enough to disable the reaction."""

    if isinstance(node, GprGene):
        return node.symbol == gene
    if node.operator == "and":
        return any(gpr_requires(child, gene) for child in node.children)
    return all(gpr_requires(child, gene) for child in node.children)


def gpr_to_text(node: GprNode) -> str:
    if isinstance(node, GprGene):
        return node.symbol
    joiner = f" {node.operator} "
    return "(" + joiner.join(gpr_to_text(child) for child in node.children) + ")"
