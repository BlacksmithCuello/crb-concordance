"""Clinical and transcriptomic record schema with the pre-specified exclusions.

Ref: Sec. 4.6 and Data availability (three sites across three regions; the
retrospective and prospective accrual targets; pCR ascertainment; the declared
exclusion criteria of insufficient RNA yield and missing outcome ascertainment).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SITES: tuple[str, ...] = ("Site A", "Site B", "Site C")
REGIONS: tuple[str, ...] = ("Region I", "Region II", "Region III")
SITE_REGION: dict[str, str] = {"Site A": "Region I", "Site B": "Region II", "Site C": "Region III"}


class Arm(str, Enum):
    RETROSPECTIVE = "retrospective"
    PROSPECTIVE = "prospective"


class Histology(str, Enum):
    ADENOCARCINOMA = "adenocarcinoma"
    MUCINOUS = "mucinous"
    SIGNET_RING = "signet-ring"


class ExclusionReason(str, Enum):
    INSUFFICIENT_RNA_YIELD = "insufficient RNA yield"
    MISSING_OUTCOME_ASCERTAINMENT = "missing outcome ascertainment"


class SchemaError(ValueError):
    """Raised when a record violates the declared schema or a design bound."""


@dataclass(frozen=True, slots=True)
class ClinicalRecord:
    """One anonymised record of the retrospective or prospective arm."""

    record_id: str
    site: str
    region: str
    arm: Arm
    age: float
    ct_stage: int
    cn_stage: int
    cea: float
    emvi: bool
    mrf: bool
    histology: Histology
    total_neoadjuvant_therapy: bool
    rna_yield_ok: bool
    outcome_ascertained: bool
    pathological_complete_response: bool | None
    mr_trg: int | None
    time_to_recurrence_months: float | None
    recurrence: bool | None
    competing_event: bool | None

    def validate(self) -> None:
        if self.site not in SITES:
            raise SchemaError(f"undeclared site: {self.site!r}")
        if self.region not in REGIONS:
            raise SchemaError(f"undeclared region: {self.region!r}")
        if SITE_REGION[self.site] != self.region:
            raise SchemaError(f"site {self.site} is not in {self.region}")
        if not 2 <= self.ct_stage <= 4:
            raise SchemaError(f"cT stage outside the declared range: {self.ct_stage!r}")
        if not 0 <= self.cn_stage <= 2:
            raise SchemaError(f"cN stage outside the declared range: {self.cn_stage!r}")
        if self.cea < 0.0:
            raise SchemaError(f"CEA must be non-negative: {self.cea!r}")
        if self.pathological_complete_response is not None and not self.outcome_ascertained:
            raise SchemaError("an ascertained outcome cannot be missing")

    def exclusion(self) -> ExclusionReason | None:
        """The first pre-specified exclusion criterion this record meets, if any."""

        if not self.rna_yield_ok:
            return ExclusionReason.INSUFFICIENT_RNA_YIELD
        if not self.outcome_ascertained:
            return ExclusionReason.MISSING_OUTCOME_ASCERTAINMENT
        return None

    def retained(self) -> bool:
        return self.exclusion() is None

    def binary_label(self) -> int | None:
        if self.pathological_complete_response is None:
            return None
        return int(self.pathological_complete_response)


@dataclass(frozen=True, slots=True)
class TranscriptomicRecord:
    """Pretreatment-biopsy expression of one retained record."""

    record_id: str
    series: str
    platform: str
    expression: tuple[tuple[str, float], ...]

    def profile(self) -> dict[str, float]:
        return dict(self.expression)

    def gene(self, symbol: str) -> float | None:
        for name, value in self.expression:
            if name == symbol:
                return value
        return None


@dataclass(frozen=True, slots=True)
class DependencyObservation:
    """One functional-genomic or pharmacological measurement for a cell line."""

    cell_line: str
    gene: str
    chronos_effect: float
    drug_response: float | None
    tissue: str
    primary_disease: str


@dataclass(frozen=True, slots=True)
class CohortCensus:
    total: int
    retained: int
    excluded: int
    by_arm: dict[str, int]
    by_site: dict[str, int]
    by_region: dict[str, int]
    pcr_positive: int
    exclusions: dict[str, int]

    @property
    def pcr_rate(self) -> float:
        if self.retained <= 0:
            return 0.0
        return self.pcr_positive / self.retained

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "retained": self.retained,
            "excluded": self.excluded,
            "by_arm": dict(sorted(self.by_arm.items())),
            "by_site": dict(sorted(self.by_site.items())),
            "by_region": dict(sorted(self.by_region.items())),
            "pcr_positive": self.pcr_positive,
            "pcr_rate": self.pcr_rate,
            "exclusions": dict(sorted(self.exclusions.items())),
        }


def census(records: tuple[ClinicalRecord, ...]) -> CohortCensus:
    by_arm: dict[str, int] = {}
    by_site: dict[str, int] = {}
    by_region: dict[str, int] = {}
    exclusions: dict[str, int] = {}
    retained = 0
    pcr_positive = 0
    for record in records:
        record.validate()
        by_arm[record.arm.value] = by_arm.get(record.arm.value, 0) + 1
        by_site[record.site] = by_site.get(record.site, 0) + 1
        by_region[record.region] = by_region.get(record.region, 0) + 1
        reason = record.exclusion()
        if reason is not None:
            exclusions[reason.value] = exclusions.get(reason.value, 0) + 1
            continue
        retained += 1
        if record.pathological_complete_response:
            pcr_positive += 1
    return CohortCensus(
        total=len(records),
        retained=retained,
        excluded=len(records) - retained,
        by_arm=by_arm,
        by_site=by_site,
        by_region=by_region,
        pcr_positive=pcr_positive,
        exclusions=exclusions,
    )


def retained_records(records: tuple[ClinicalRecord, ...]) -> tuple[ClinicalRecord, ...]:
    return tuple(record for record in records if record.retained())


def labelled_records(records: tuple[ClinicalRecord, ...]) -> tuple[ClinicalRecord, ...]:
    return tuple(
        record for record in retained_records(records) if record.binary_label() is not None
    )
