"""Site-stratified survival analysis with competing-risk handling.

Ref: Sec. 4.6 (time-to-event outcomes use Cox models stratified by site with
competing-risk handling for local recurrence, and because records cluster within
sites the cross-site clinical models include a site-level random effect).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.stats.delong import two_sided_normal_p

MAX_ITERATIONS = 50
CONVERGENCE_TOLERANCE = 1e-9
RIDGE = 1e-8


class SurvivalError(ValueError):
    """Raised when a survival model cannot be fitted."""


@dataclass(frozen=True, slots=True)
class CoxResult:
    """Fitted coefficients with their standard errors and fit statistics."""

    names: tuple[str, ...]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    log_likelihood: float
    iterations: int
    events: int
    concordance: float

    def z_scores(self) -> np.ndarray:
        return self.coefficients / np.where(self.standard_errors > 0, self.standard_errors, 1.0)

    def p_values(self) -> np.ndarray:
        return np.asarray([two_sided_normal_p(float(z)) for z in self.z_scores()], dtype=float)

    def as_rows(self) -> list[dict[str, float | str]]:
        intervals = self.confidence_intervals()
        return [
            {
                "name": self.names[index],
                "coefficient": float(self.coefficients[index]),
                "standard_error": float(self.standard_errors[index]),
                "hazard_ratio": float(np.exp(self.coefficients[index])),
                "ci_lower": float(np.exp(intervals[index][0])),
                "ci_upper": float(np.exp(intervals[index][1])),
                "p_value": float(self.p_values()[index]),
            }
            for index in range(len(self.names))
        ]

    def confidence_intervals(self, *, level: float = 0.95) -> tuple[tuple[float, float], ...]:
        from crb_concordance.utils.numerics import normal_ppf

        critical = normal_ppf(1.0 - (1.0 - level) / 2.0)
        return tuple(
            (
                float(self.coefficients[index] - critical * self.standard_errors[index]),
                float(self.coefficients[index] + critical * self.standard_errors[index]),
            )
            for index in range(len(self.names))
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.as_rows(),
            "log_likelihood": self.log_likelihood,
            "iterations": self.iterations,
            "events": self.events,
            "concordance": self.concordance,
        }


def _risk_sets(times: np.ndarray, strata: np.ndarray) -> list[np.ndarray]:
    event_times = np.unique(times)
    sets: list[np.ndarray] = []
    for stratum in np.unique(strata):
        in_stratum = strata == stratum
        for time in event_times:
            at_risk = in_stratum & (times >= time)
            if np.any(at_risk):
                sets.append(np.flatnonzero(at_risk))
    return sets


def _partial_likelihood(
    covariates: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    strata: np.ndarray,
    beta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    linear = covariates @ beta
    exp_linear = np.exp(np.clip(linear, -50.0, 50.0))
    log_likelihood = 0.0
    gradient = np.zeros_like(beta)
    information = np.zeros((beta.size, beta.size))
    for stratum in np.unique(strata):
        in_stratum = strata == stratum
        for time in np.unique(times[in_stratum & (events > 0)]):
            events_here = in_stratum & (times == time) & (events > 0)
            at_risk = in_stratum & (times >= time)
            if not np.any(events_here) or not np.any(at_risk):
                continue
            weight = exp_linear[at_risk]
            total = float(np.sum(weight))
            if total <= 0.0:
                raise SurvivalError("risk-set weight vanished; rescale the covariates")
            covariates_risk = covariates[at_risk]
            weighted_mean = (weight[:, None] * covariates_risk).sum(axis=0) / total
            count = float(np.count_nonzero(events_here))
            sum_events = covariates[events_here].sum(axis=0)
            log_likelihood += float(sum_events @ beta) - count * float(np.log(total))
            gradient += sum_events - count * weighted_mean
            second = (weight[:, None] * covariates_risk).T @ covariates_risk / total
            information += count * (second - np.outer(weighted_mean, weighted_mean))
    return log_likelihood, gradient, information


def concordance_index(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    comparable = 0
    concordant = 0.0
    for left in range(times.size):
        if events[left] == 0:
            continue
        for right in range(times.size):
            if times[right] <= times[left]:
                continue
            comparable += 1
            if risk[left] > risk[right]:
                concordant += 1.0
            elif abs(risk[left] - risk[right]) <= 1e-12:
                concordant += 0.5
    if comparable == 0:
        return 0.0
    return concordant / comparable


def fit_cox(
    times: np.ndarray,
    events: np.ndarray,
    covariates: np.ndarray,
    *,
    names: tuple[str, ...],
    strata: np.ndarray | None = None,
) -> CoxResult:
    """Fit a (optionally site-stratified) Cox model by Newton-Raphson."""

    durations = np.asarray(times, dtype=float)
    observed = np.asarray(events, dtype=int)
    design = np.asarray(covariates, dtype=float)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if durations.shape[0] != design.shape[0] or observed.shape[0] != design.shape[0]:
        raise SurvivalError("times, events and covariates must share a row count")
    if design.shape[1] != len(names):
        raise SurvivalError("one covariate name is required per column")
    if durations.size == 0:
        raise SurvivalError("the survival table is empty")
    if not np.any(observed > 0):
        raise SurvivalError("the survival table carries no events")
    groups = (
        np.zeros(durations.size, dtype=int) if strata is None else np.asarray(strata, dtype=int)
    )
    beta = np.zeros(design.shape[1])
    previous = -np.inf
    iterations = 0
    for step in range(MAX_ITERATIONS):
        iterations = step + 1
        log_likelihood, gradient, information = _partial_likelihood(
            design, durations, observed, groups, beta
        )
        if abs(log_likelihood - previous) <= CONVERGENCE_TOLERANCE:
            break
        previous = log_likelihood
        updated = information + RIDGE * np.eye(beta.size)
        beta = beta + np.linalg.solve(updated, gradient)
    log_likelihood, _, information = _partial_likelihood(design, durations, observed, groups, beta)
    covariance = np.linalg.inv(information + RIDGE * np.eye(beta.size))
    return CoxResult(
        names=names,
        coefficients=beta,
        standard_errors=np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        log_likelihood=float(log_likelihood),
        iterations=iterations,
        events=int(np.count_nonzero(observed > 0)),
        concordance=concordance_index(durations, observed, design @ beta),
    )


@dataclass(frozen=True, slots=True)
class CompetingRiskFit:
    """Cause-specific hazards for the event of interest and the competing event."""

    of_interest: CoxResult
    competing: CoxResult
    competing_events: int

    def as_dict(self) -> dict[str, object]:
        return {
            "of_interest": self.of_interest.as_dict(),
            "competing": self.competing.as_dict(),
            "competing_events": self.competing_events,
        }


def fit_competing_risks(
    times: np.ndarray,
    event_type: np.ndarray,
    covariates: np.ndarray,
    *,
    names: tuple[str, ...],
    of_interest: int = 1,
    strata: np.ndarray | None = None,
) -> CompetingRiskFit:
    """Cause-specific Cox fits, treating the other cause as censoring."""

    types = np.asarray(event_type, dtype=int)
    design = np.asarray(covariates, dtype=float)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    interest = (types == of_interest).astype(int)
    competing = ((types != 0) & (types != of_interest)).astype(int)
    if int(competing.sum()) == 0:
        raise SurvivalError("the competing event never occurs in this table")
    return CompetingRiskFit(
        of_interest=fit_cox(times, interest, design, names=names, strata=strata),
        competing=fit_cox(times, competing, design, names=names, strata=strata),
        competing_events=int(competing.sum()),
    )


def cumulative_incidence(
    times: np.ndarray, event_type: np.ndarray, *, of_interest: int = 1
) -> np.ndarray:
    """Aalen-Johansen cumulative incidence of one cause in the presence of another."""

    durations = np.asarray(times, dtype=float)
    types = np.asarray(event_type, dtype=int)
    if durations.size == 0:
        raise SurvivalError("the survival table is empty")
    horizon = np.unique(durations)
    overall_survival = 1.0
    incidence = np.zeros(horizon.size)
    for position, time in enumerate(horizon):
        at_risk = int(np.count_nonzero(durations >= time))
        if at_risk == 0:
            break
        deaths = int(np.count_nonzero((durations == time) & (types > 0)))
        cause = int(np.count_nonzero((durations == time) & (types == of_interest)))
        if deaths > 0:
            hazard = deaths / at_risk
            incidence[position] = overall_survival * (cause / at_risk)
            overall_survival *= 1.0 - hazard
        else:
            incidence[position] = 0.0
    return np.cumsum(incidence)


def site_gap_summary(
    per_site_aucs: tuple[float, ...], *, flag_threshold: float = 0.10
) -> dict[str, float]:
    """Inter-site gap of the primary discrimination metric against the declared flag."""

    values = np.asarray(per_site_aucs, dtype=float)
    if values.size < 2:
        raise SurvivalError("a site gap needs at least two sites")
    gap = float(np.max(values) - np.min(values))
    return {
        "gap": gap,
        "flag_threshold": flag_threshold,
        "flagged": 1.0 if gap > flag_threshold else 0.0,
    }
