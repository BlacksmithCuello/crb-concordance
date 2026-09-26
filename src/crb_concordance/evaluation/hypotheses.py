"""Assembly of every pre-specified hypothesis into one reportable object.

Ref: Sec. 4.1 (Propositions 1 and 2); Sec. 4.4 (the recall@10 falsification
inequality and the parity principle); Sec. 4.5 (H1 on the fusion rule and H2 on the
conflict trace); Sec. 4.6 (prospective sizing and the design identities).
"""

from __future__ import annotations

from dataclasses import dataclass

from crb_concordance.belief.discount import DiscountRates
from crb_concordance.belief.guarantees import (
    GuaranteeCheck,
    conflict_monotonicity_scan,
    pignistic_bias_floor,
    proposition_summary,
    zero_conflict_limit,
)
from crb_concordance.simulation.contrasts import (
    HypothesisOutcome,
    absent_evidence_widens_intervals,
    conflict_monotonicity_across_cells,
    cutoff_reachability,
    falsification_inequality,
    falsification_inequality_saturated,
)
from crb_concordance.simulation.runner import SimulationResult
from crb_concordance.stats.power import SizingResult, accrual_range_covers
from crb_concordance.utils.types import Verdict


@dataclass(frozen=True, slots=True)
class HypothesisReport:
    """Every hypothesis with its verdict, plus the design identities."""

    propositions: dict[str, GuaranteeCheck]
    simulation: dict[str, HypothesisOutcome]
    reachability: dict[str, float]
    sizing: dict[str, object]
    floor: float
    zero_conflict_limits: dict[str, float]

    def outcomes(self) -> dict[str, str]:
        combined = {name: check.holds for name, check in self.propositions.items()}
        combined.update(
            {name: outcome.verdict is Verdict.PASS for name, outcome in self.simulation.items()}
        )
        return {
            name: (Verdict.PASS.value if holds else Verdict.FAIL.value)
            for name, holds in combined.items()
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "propositions": {
                name: {"holds": check.holds, "detail": check.detail}
                for name, check in self.propositions.items()
            },
            "simulation": {name: outcome.as_dict() for name, outcome in self.simulation.items()},
            "reachability": self.reachability,
            "sizing": self.sizing,
            "ignorance_floor": self.floor,
            "zero_conflict_limits": self.zero_conflict_limits,
            "outcomes": self.outcomes(),
        }


def build_hypothesis_report(
    rates: DiscountRates,
    simulation: SimulationResult,
    sizing: SizingResult,
) -> HypothesisReport:
    """Evaluate every pre-specified claim that this release can execute."""

    propositions = proposition_summary(rates)
    outcomes = {
        "recall10_parity_inequality": falsification_inequality(simulation),
        "recall_parity_inequality_at_saturated_cutoff": falsification_inequality_saturated(
            simulation
        ),
        "conflict_mass_increases_with_declared_conflict": conflict_monotonicity_across_cells(
            simulation
        ),
        "interval_width_increases_with_absent_evidence": absent_evidence_widens_intervals(
            simulation
        ),
    }
    uppers, lowers = zero_conflict_limit(rates, target=True), zero_conflict_limit(
        rates, target=False
    )
    return HypothesisReport(
        propositions=propositions,
        simulation=outcomes,
        reachability=cutoff_reachability(simulation),
        sizing={
            "rows": sizing.as_rows(),
            "accrual_coverage": accrual_range_covers(sizing),
        },
        floor=pignistic_bias_floor(rates),
        zero_conflict_limits={"target_v": uppers[0], "target_not_v": lowers[0]},
    )


def monotonicity_detail(rates: DiscountRates) -> dict[str, object]:
    scan = conflict_monotonicity_scan(rates)
    return {
        "pressure_points": int(scan.pressure.size),
        "width_start": float(scan.width[0]),
        "width_end": float(scan.width[-1]),
        "conflict_start": float(scan.total_conflict[0]),
        "conflict_end": float(scan.total_conflict[-1]),
        "worst_decrease": scan.worst_decrease(),
        "monotone": scan.monotone_non_decreasing(),
    }
