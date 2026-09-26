"""Marginal logistic model for the reader study.

Ref: Sec. 4.6 (the reader-study team-arm gain is tested by a mixed-effects logistic
generalised-estimating-equations model of per-case agreement with eventual outcome,
with reader as a random effect and arm as a fixed effect, reporting the assisted-arm
coefficient's 95% confidence interval and two-sided p-value, cross-checked by a
reader-level cluster bootstrap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crb_concordance.stats.delong import two_sided_normal_p
from crb_concordance.utils.numerics import normal_ppf

MAX_ITERATIONS = 100
CONVERGENCE_TOLERANCE = 1e-10
RIDGE = 1e-8


class GeeError(ValueError):
    """Raised when a marginal model cannot be fitted."""


@dataclass(frozen=True, slots=True)
class GeeResult:
    """GEE coefficients with robust (sandwich) standard errors."""

    names: tuple[str, ...]
    coefficients: np.ndarray
    robust_standard_errors: np.ndarray
    working_correlation: float
    iterations: int
    clusters: int
    observations: int

    def p_values(self) -> np.ndarray:
        return np.asarray(
            [
                two_sided_normal_p(
                    float(self.coefficients[i] / self.robust_standard_errors[i])
                    if self.robust_standard_errors[i] > 0
                    else 0.0
                )
                for i in range(len(self.names))
            ],
            dtype=float,
        )

    def interval(self, name: str, *, level: float = 0.95) -> tuple[float, float]:
        try:
            index = self.names.index(name)
        except ValueError as error:
            raise GeeError(f"unknown coefficient: {name}") from error
        critical = normal_ppf(1.0 - (1.0 - level) / 2.0)
        half = critical * float(self.robust_standard_errors[index])
        return (
            float(self.coefficients[index] - half),
            float(self.coefficients[index] + half),
        )

    def as_dict(self) -> dict[str, object]:
        p_values = self.p_values()
        rows = [
            {
                "name": self.names[index],
                "coefficient": float(self.coefficients[index]),
                "robust_standard_error": float(self.robust_standard_errors[index]),
                "odds_ratio": float(np.exp(self.coefficients[index])),
                "p_value": float(p_values[index]),
            }
            for index in range(len(self.names))
        ]
        return {
            "rows": rows,
            "working_correlation": self.working_correlation,
            "iterations": self.iterations,
            "clusters": self.clusters,
            "observations": self.observations,
        }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _working_matrix(design: np.ndarray, cluster: np.ndarray, correlation: float) -> np.ndarray:
    size = design.shape[0]
    matrix = np.eye(size)
    for identifier in np.unique(cluster):
        members = np.flatnonzero(cluster == identifier)
        for left in members:
            for right in members:
                if left != right:
                    matrix[left, right] = correlation
    return matrix


def fit_gee_logistic(
    response: np.ndarray,
    design: np.ndarray,
    cluster: np.ndarray,
    *,
    names: tuple[str, ...],
) -> GeeResult:
    """Exchangeable-working-correlation marginal logistic fit with sandwich errors."""

    y = np.asarray(response, dtype=float)
    x = np.asarray(design, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    groups = np.asarray(cluster)
    if y.shape[0] != x.shape[0] or y.shape[0] != groups.shape[0]:
        raise GeeError("response, design and cluster must share a row count")
    if x.shape[1] != len(names):
        raise GeeError("one coefficient name is required per column")
    if np.unique(y).size < 2:
        raise GeeError("the reader study needs both agreement outcomes")
    identifiers = np.unique(groups)
    if identifiers.size < 2:
        raise GeeError("at least two readers are required")
    beta = np.zeros(x.shape[1])
    correlation = 0.0
    iterations = 0
    for step in range(MAX_ITERATIONS):
        iterations = step + 1
        linear = x @ beta
        mean = _sigmoid(linear)
        variance = np.clip(mean * (1.0 - mean), 1e-8, None)
        working = _working_matrix(x, groups, correlation)
        weighted = np.diag(np.sqrt(variance)) @ working @ np.diag(np.sqrt(variance))
        try:
            bread = np.linalg.inv(x.T @ np.linalg.solve(weighted, x) + RIDGE * np.eye(beta.size))
        except np.linalg.LinAlgError as error:
            raise GeeError("working correlation matrix is singular") from error
        update = bread @ (x.T @ np.linalg.solve(weighted, y - mean))
        beta = beta + update
        if float(np.max(np.abs(update))) <= CONVERGENCE_TOLERANCE:
            break
        correlation = _estimate_correlation(y, mean, groups)
    linear = x @ beta
    mean = _sigmoid(linear)
    variance = np.clip(mean * (1.0 - mean), 1e-8, None)
    working = _working_matrix(x, groups, correlation)
    weighted = np.diag(np.sqrt(variance)) @ working @ np.diag(np.sqrt(variance))
    bread = np.linalg.inv(x.T @ np.linalg.solve(weighted, x) + RIDGE * np.eye(beta.size))
    residual = y - mean
    meat = np.zeros((beta.size, beta.size))
    for identifier in identifiers:
        members = np.flatnonzero(groups == identifier)
        block = x[members].T @ residual[members]
        meat += np.outer(block, block)
    sandwich = bread @ meat @ bread
    return GeeResult(
        names=names,
        coefficients=beta,
        robust_standard_errors=np.sqrt(np.clip(np.diag(sandwich), 0.0, None)),
        working_correlation=correlation,
        iterations=iterations,
        clusters=int(identifiers.size),
        observations=int(y.shape[0]),
    )


def _estimate_correlation(response: np.ndarray, mean: np.ndarray, cluster: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for identifier in np.unique(cluster):
        members = np.flatnonzero(cluster == identifier)
        if members.size < 2:
            continue
        for left in members:
            for right in members:
                if left == right:
                    continue
                numerator += float((response[left] - mean[left]) * (response[right] - mean[right]))
                denominator += float(
                    np.sqrt(
                        max(mean[left] * (1.0 - mean[left]), 1e-8)
                        * max(mean[right] * (1.0 - mean[right]), 1e-8)
                    )
                )
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 0.95))


def reader_study_design(
    per_case: np.ndarray, readers: np.ndarray, arms: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Design matrix with an intercept, the arm fixed effect and reader indicators."""

    y = np.asarray(per_case, dtype=float)
    reader = np.asarray(readers)
    arm = np.asarray(arms, dtype=float)
    if y.shape[0] != reader.shape[0] or y.shape[0] != arm.shape[0]:
        raise GeeError("per-case responses, readers and arms must share a length")
    identifiers = sorted({str(value) for value in np.unique(reader)})
    columns = [np.ones(y.shape[0]), arm]
    names = ["intercept", "assisted_arm"]
    for identifier in identifiers[1:]:
        columns.append((np.asarray([str(value) for value in reader]) == identifier).astype(float))
        names.append(f"reader={identifier}")
    return np.column_stack(columns), tuple(names)


def arm_gain_report(result: GeeResult, *, arm_name: str = "assisted_arm") -> dict[str, object]:
    """The assisted-arm coefficient with its interval, as the protocol reports it."""

    lower, upper = result.interval(arm_name)
    try:
        index = result.names.index(arm_name)
    except ValueError as error:
        raise GeeError(f"unknown coefficient: {arm_name}") from error
    return {
        "coefficient": float(result.coefficients[index]),
        "ci_lower": lower,
        "ci_upper": upper,
        "odds_ratio": float(np.exp(result.coefficients[index])),
        "p_value": float(result.p_values()[index]),
        "readers": result.clusters,
        "observations": result.observations,
        "working_correlation": result.working_correlation,
    }
