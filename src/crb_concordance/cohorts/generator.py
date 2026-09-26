"""Schema-compatible generators for the cohorts and the simulation design.

Ref: Sec. 4.6 (three sites across three regions; retrospective 3,000-3,600 records
at about 1,000-1,200 per site; prospective target accrual about 750-900; pCR rate
25% within the reported 20-30% range; baseline standard-of-care AUROC 0.70 within
the reported 0.65-0.74 range; reader panel of at least 8 readers on at least 155
shared cases); Supplementary Table S1 (2,000 simulated candidates per seed, 15%
ground-truth prevalence, conflict-probability grid and absent-evidence grid).

The clinical cohorts are held at their institutions and are not redistributed, so
this module generates records that carry exactly the declared design marginals and
marks every cohort-level quantity as a simulation. The evidence generator is the
object the simulation study quantifies over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.cohorts.schema import (
    SITE_REGION,
    SITES,
    Arm,
    ClinicalRecord,
    Histology,
    TranscriptomicRecord,
)
from crb_concordance.utils.numerics import rng_from
from crb_concordance.utils.types import MODALITY_ORDER, Modality

DESIGN_SEED = 20260101
CANDIDATE_SIMULATION_SEED = 1


@dataclass(frozen=True, slots=True)
class CohortDesign:
    """Every design constant the manuscript fixes for the clinical arms."""

    retrospective_site_counts: tuple[int, int, int] = (1000, 1200, 1100)
    prospective_size: int = 825
    pcr_rate: float = 0.25
    soc_auroc: float = 0.70
    rna_yield_failure_rate: float = 0.06
    outcome_missing_rate: float = 0.03
    reader_count: int = 8
    shared_cases: int = 155
    seed: int = DESIGN_SEED

    @property
    def retrospective_size(self) -> int:
        return int(sum(self.retrospective_site_counts))

    def validate(self) -> None:
        if not 3000 <= self.retrospective_size <= 3600:
            raise ValueError("retrospective accrual outside the declared 3,000-3,600 range")
        if not 750 <= self.prospective_size <= 900:
            raise ValueError("prospective accrual outside the declared 750-900 range")
        if not 0.20 <= self.pcr_rate <= 0.30:
            raise ValueError("pCR rate outside the reported 20-30% range")
        if not 0.65 <= self.soc_auroc <= 0.74:
            raise ValueError("baseline AUROC outside the reported 0.65-0.74 range")
        if len(self.retrospective_site_counts) != len(SITES):
            raise ValueError("one accrual count is required per site")

    def as_dict(self) -> dict[str, object]:
        return {
            "retrospective_site_counts": list(self.retrospective_site_counts),
            "retrospective_size": self.retrospective_size,
            "prospective_size": self.prospective_size,
            "pcr_rate": self.pcr_rate,
            "soc_auroc": self.soc_auroc,
            "rna_yield_failure_rate": self.rna_yield_failure_rate,
            "outcome_missing_rate": self.outcome_missing_rate,
            "reader_count": self.reader_count,
            "shared_cases": self.shared_cases,
            "seed": self.seed,
        }


def _record(
    index: int,
    site: str,
    arm: Arm,
    pcr: bool,
    generator: np.random.Generator,
    design: CohortDesign,
) -> ClinicalRecord:
    histology = Histology.ADENOCARCINOMA
    draw = generator.random()
    if draw > 0.94:
        histology = Histology.SIGNET_RING
    elif draw > 0.86:
        histology = Histology.MUCINOUS
    ascertained = bool(generator.random() >= design.outcome_missing_rate)
    pcr = bool(pcr) if ascertained else None
    mr_trg = (
        None if (not ascertained or pcr) else int(generator.choice([3, 4, 5], p=[0.45, 0.35, 0.20]))
    )
    recurrence = None
    time_to_recurrence = None
    competing = None
    if arm is Arm.PROSPECTIVE:
        recurrence = bool(generator.random() < (0.12 if pcr else 0.34))
        time_to_recurrence = float(np.round(generator.uniform(3.0, 36.0), 3))
        competing = bool(generator.random() < 0.05)
    return ClinicalRecord(
        record_id=f"{arm.value[:4].upper()}-{site[-1]}-{index:05d}",
        site=site,
        region=SITE_REGION[site],
        arm=arm,
        age=float(np.round(generator.normal(63.0, 10.0), 1)),
        ct_stage=int(generator.choice([2, 3, 4], p=[0.08, 0.72, 0.20])),
        cn_stage=int(generator.choice([0, 1, 2], p=[0.34, 0.48, 0.18])),
        cea=float(np.round(np.abs(generator.normal(6.0, 4.5)) + 1.0, 2)),
        emvi=bool(generator.random() < 0.42),
        mrf=bool(generator.random() < 0.18),
        histology=histology,
        total_neoadjuvant_therapy=bool(generator.random() < 0.55),
        rna_yield_ok=bool(generator.random() >= design.rna_yield_failure_rate),
        outcome_ascertained=ascertained,
        pathological_complete_response=pcr,
        mr_trg=mr_trg,
        time_to_recurrence_months=time_to_recurrence,
        recurrence=recurrence,
        competing_event=competing,
    )


def generate_cohort(design: CohortDesign) -> tuple[ClinicalRecord, ...]:
    """Generate the retrospective and prospective arms at the declared marginals."""

    design.validate()
    generator = rng_from(design.seed)
    records: list[ClinicalRecord] = []
    counter = 0
    for site, count in zip(SITES, design.retrospective_site_counts, strict=True):
        positives = int(round(count * design.pcr_rate))
        flags = [True] * positives + [False] * (count - positives)
        generator.shuffle(flags)
        for flag in flags:
            counter += 1
            records.append(_record(counter, site, Arm.RETROSPECTIVE, bool(flag), generator, design))
    per_site = design.prospective_size // len(SITES)
    remainder = design.prospective_size - per_site * len(SITES)
    for position, site in enumerate(SITES):
        count = per_site + (1 if position < remainder else 0)
        positives = int(round(count * design.pcr_rate))
        flags = [True] * positives + [False] * (count - positives)
        generator.shuffle(flags)
        for flag in flags:
            counter += 1
            records.append(_record(counter, site, Arm.PROSPECTIVE, bool(flag), generator, design))
    return tuple(records)


def generate_expression(
    records: tuple[ClinicalRecord, ...],
    genes: tuple[str, ...],
    *,
    seed: int = DESIGN_SEED,
    series: str = "GSE35452",
    platform: str = "Affymetrix HG-U133 Plus 2.0",
) -> dict[str, TranscriptomicRecord]:
    """Expression stand-in whose per-gene values track the response label."""

    generator = rng_from(seed)
    profiles: dict[str, TranscriptomicRecord] = {}
    for record in records:
        if not record.retained():
            continue
        label = record.binary_label()
        if label is None:
            continue
        values: list[tuple[str, float]] = []
        for position, gene in enumerate(genes):
            shift = 0.45 if (label == 1 and position % 3 == 0) else 0.0
            values.append((gene, float(np.round(generator.normal(shift, 1.0), 6))))
        profiles[record.record_id] = TranscriptomicRecord(
            record_id=record.record_id,
            series=series,
            platform=platform,
            expression=tuple(values),
        )
    return profiles


@dataclass(frozen=True, slots=True)
class EvidenceDesign:
    """The simulation grid of Supplementary Table S1."""

    n_candidates: int = 2000
    prevalence: float = 0.15
    conflict_probability: float = 0.3
    absent_fraction: float = 0.0
    seed: int = CANDIDATE_SIMULATION_SEED
    support_concentration: float = 4.0
    contradict_concentration: float = 2.5

    def validate(self) -> None:
        if self.n_candidates < 1:
            raise ValueError("the candidate pool must be non-empty")
        if not 0.0 < self.prevalence < 1.0:
            raise ValueError("ground-truth prevalence must lie in (0, 1)")
        if not 0.0 <= self.conflict_probability <= 1.0:
            raise ValueError("conflict probability must lie in [0, 1]")
        if not 0.0 <= self.absent_fraction <= 1.0:
            raise ValueError("absent-evidence fraction must lie in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "n_candidates": self.n_candidates,
            "prevalence": self.prevalence,
            "conflict_probability": self.conflict_probability,
            "absent_fraction": self.absent_fraction,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class SimulatedCandidate:
    """One synthetic candidate with its hidden status and per-modality evidence."""

    gene: str
    vulnerable: bool
    scores: dict[Modality, float | None]

    def available(self) -> tuple[Modality, ...]:
        return tuple(modality for modality in MODALITY_ORDER if self.scores[modality] is not None)

    def supports(self, modality: Modality, *, threshold: float = 0.5) -> bool | None:
        score = self.scores[modality]
        if score is None:
            return None
        return score >= threshold


def simulate_candidates(design: EvidenceDesign) -> tuple[SimulatedCandidate, ...]:
    """Draw candidates whose modality evidence conflicts at the declared rate."""

    design.validate()
    generator = rng_from(design.seed)
    n_vulnerable = int(round(design.n_candidates * design.prevalence))
    flags = np.concatenate(
        [
            np.ones(n_vulnerable, dtype=bool),
            np.zeros(design.n_candidates - n_vulnerable, dtype=bool),
        ]
    )
    generator.shuffle(flags)
    strength = design.support_concentration
    weakness = design.contradict_concentration
    candidates: list[SimulatedCandidate] = []
    for position, vulnerable in enumerate(flags):
        scores: dict[Modality, float | None] = {}
        for modality in MODALITY_ORDER:
            if generator.random() < design.absent_fraction:
                scores[modality] = None
                continue
            agrees = generator.random() >= design.conflict_probability
            supports_v = bool(vulnerable) == bool(agrees)
            if supports_v:
                scores[modality] = float(generator.beta(strength, 1.5))
            else:
                scores[modality] = float(generator.beta(1.5, weakness))
        candidates.append(
            SimulatedCandidate(gene=f"G{position:05d}", vulnerable=bool(vulnerable), scores=scores)
        )
    return tuple(candidates)


def simulate_candidate_pool(
    design: EvidenceDesign, *, controls: tuple[str, ...] = ()
) -> tuple[SimulatedCandidate, ...]:
    """The synthetic pool prefixed with the named controls, whose status is declared."""

    drawn = list(simulate_candidates(design))
    for position, symbol in enumerate(controls):
        drawn.insert(
            position,
            SimulatedCandidate(
                gene=symbol,
                vulnerable=symbol in {"SLC2A1"},
                scores=dict.fromkeys(MODALITY_ORDER, None),
            ),
        )
    return tuple(drawn)
