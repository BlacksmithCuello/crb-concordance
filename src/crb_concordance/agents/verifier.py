"""Critic/verifier: discounting, combination, trace and structural checking.

Ref: Sec. 4.2 item (5) (the critic/verifier agent performs the discounting, the
combination, and the report generation, and checks every output for structural
correctness against the baselines and ablations); Sec. 4.1 (belief, plausibility and
pignistic quantities); Supplementary Table S1 (the conflict-trace flagging
threshold is a planning value fixed at implementation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crb_concordance.agents.base import RawEvidence
from crb_concordance.agents.mapping import MassMapper
from crb_concordance.belief.combine import CombinationResult, combine_modality_masses
from crb_concordance.belief.discount import DiscountRates, discount
from crb_concordance.belief.mass import AgentMass, BeliefReport
from crb_concordance.belief.trace import PLANNING_FLAG_THRESHOLD, ConflictTrace, build_trace
from crb_concordance.discovery.ledger import ProvenanceLedger
from crb_concordance.utils.types import MODALITY_ORDER, Modality

STRUCTURAL_TOLERANCE = 1e-9


class VerificationError(ValueError):
    """Raised when a combined belief object fails a structural invariant."""


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """The verifier's output for one candidate: interval, trace and ledger rows."""

    candidate: str
    masses: dict[Modality, AgentMass]
    combined: CombinationResult
    trace: ConflictTrace
    report: BeliefReport
    ledger_rows: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "mass_triples": {
                modality.value: mass.triple.as_tuple() for modality, mass in self.masses.items()
            },
            "discount_rates": {
                modality.value: mass.discount_rate for modality, mass in self.masses.items()
            },
            "belief": self.report.belief,
            "plausibility": self.report.plausibility,
            "pignistic": self.report.pignistic,
            "interval_width": self.report.interval_width,
            "total_conflict": self.combined.total_conflict,
            "flagged_steps": len(self.trace.flagged()),
            "trace": self.trace.as_rows(),
        }


@dataclass(slots=True)
class Verifier:
    """Applies Eq. (2), Algorithm 2 and the audit checks."""

    rates: DiscountRates
    mapper: MassMapper
    conflict_threshold: float = PLANNING_FLAG_THRESHOLD
    calibration_fold: int = 0
    checks: list[str] = field(default_factory=list)

    def combine_candidate(
        self,
        candidate: str,
        evidences: dict[Modality, RawEvidence],
        *,
        ledger: ProvenanceLedger | None = None,
        agent_names: dict[Modality, str] | None = None,
    ) -> VerificationOutcome:
        missing = [modality for modality in MODALITY_ORDER if modality not in evidences]
        if missing:
            raise VerificationError(f"missing evidence for {[m.value for m in missing]}")
        masses: dict[Modality, AgentMass] = {}
        for modality in MODALITY_ORDER:
            evidence = evidences[modality]
            triple = self.mapper.map_evidence(evidence)
            rate = self.rates.of(modality)
            discounted = discount(triple, rate)
            masses[modality] = AgentMass(
                modality=modality,
                triple=discounted,
                discount_rate=rate,
                calibration_fold=self.calibration_fold,
                evidence_available=evidence.present,
            )
        order = self.rates.reliability_order()
        combined = combine_modality_masses(
            {modality: mass.triple for modality, mass in masses.items()}, order
        )
        trace = build_trace(combined, self.conflict_threshold)
        report = BeliefReport(
            candidate=candidate,
            belief=combined.belief,
            plausibility=combined.plausibility,
            pignistic=combined.pignistic,
            total_conflict=combined.total_conflict,
            trace=combined.trace_rows(),
        )
        self.audit(report, masses, combined)
        rows: tuple[int, ...] = ()
        if ledger is not None:
            names = agent_names or {}
            first = len(ledger)
            for modality in MODALITY_ORDER:
                ledger.record(
                    candidate=candidate,
                    agent=names.get(modality, modality.value),
                    modality=modality,
                    evidence=evidences[modality],
                    mass=masses[modality],
                )
            rows = tuple(range(first, len(ledger)))
        return VerificationOutcome(
            candidate=candidate,
            masses=masses,
            combined=combined,
            trace=trace,
            report=report,
            ledger_rows=rows,
        )

    def audit(
        self,
        report: BeliefReport,
        masses: dict[Modality, AgentMass],
        combined: CombinationResult,
    ) -> None:
        for modality, mass in masses.items():
            total = sum(mass.triple.as_tuple())
            if abs(total - 1.0) > 1e-7:
                raise VerificationError(
                    f"{report.candidate}/{modality.value}: discounted mass does not sum to one ({total})"
                )
        if (
            not report.belief - STRUCTURAL_TOLERANCE
            <= report.pignistic
            <= report.plausibility + STRUCTURAL_TOLERANCE
        ):
            raise VerificationError(
                f"{report.candidate}: pignistic value outside the belief interval"
            )
        if abs(report.interval_width - combined.triple.theta) > 1e-9:
            raise VerificationError(
                f"{report.candidate}: interval width is not the uncommitted mass"
            )
        if combined.total_conflict < -1e-12:
            raise VerificationError(f"{report.candidate}: negative accumulated conflict")

    def structural_report(self, outcomes: tuple[VerificationOutcome, ...]) -> dict[str, object]:
        widths = [outcome.report.interval_width for outcome in outcomes]
        pignistics = [outcome.report.pignistic for outcome in outcomes]
        conflicts = [outcome.combined.total_conflict for outcome in outcomes]
        return {
            "candidates": len(outcomes),
            "all_triples_normalised": all(
                abs(sum(mass.triple.as_tuple()) - 1.0) <= 1e-7
                for outcome in outcomes
                for mass in outcome.masses.values()
            ),
            "all_pignistic_inside_interval": all(
                outcome.report.belief <= outcome.report.pignistic <= outcome.report.plausibility
                for outcome in outcomes
            ),
            "mean_interval_width": sum(widths) / len(widths) if widths else 0.0,
            "mean_pignistic": sum(pignistics) / len(pignistics) if pignistics else 0.0,
            "mean_conflict": sum(conflicts) / len(conflicts) if conflicts else 0.0,
            "combination_order": [modality.value for modality in self.rates.reliability_order()],
        }
