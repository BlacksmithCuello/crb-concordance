"""Report assembly and the plain-text rendering of every executed study.

Ref: Table 1 and Table 2 (every outcome cell of the pre-specified benchmarks is
pending until the cohort arms are analysed); Sec. 2 (the in silico simulation is the
one study carried out, so it is the only one whose cells carry values); Sec. 4.6
(calibration is summarised by the expected calibration error and the Brier score,
defined identically at discovery time and at clinical time).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crb_concordance.benchmark.registry import (
    CONSTRAINT_COLUMNS,
    FUSION_PROPOSED,
    PRIMARY_COLUMNS,
    ROWS,
    validate_registry,
)
from crb_concordance.utils.atomic import atomic_write_text
from crb_concordance.utils.types import Verdict

PENDING = "[pending]"


@dataclass(frozen=True, slots=True)
class TableCell:
    """One outcome cell, either pending or carrying an executed value."""

    value: str
    executed: bool

    def as_dict(self) -> dict[str, object]:
        return {"value": self.value, "executed": self.executed}


def pending_row(name: str, *, family: str, reference: int | None) -> dict[str, object]:
    return {
        "method": name,
        "family": family,
        "reference": reference,
        "primary": dict.fromkeys(PRIMARY_COLUMNS, PENDING),
        "constraint": dict.fromkeys(CONSTRAINT_COLUMNS, PENDING),
        "executed": False,
    }


def table_one_snapshot(
    executed: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    """Table 1 with pending cells and the executed rows substituted where available."""

    registry = validate_registry()
    values = executed or {}
    rows: list[dict[str, object]] = []
    for row in ROWS:
        payload = pending_row(row.name, family=row.family.value, reference=row.reference)
        if row.name in values:
            payload["primary"] = {
                column: values[row.name].get(column, PENDING) for column in PRIMARY_COLUMNS
            }
            payload["constraint"] = {
                column: values[row.name].get(column, PENDING) for column in CONSTRAINT_COLUMNS
            }
            payload["executed"] = True
        rows.append(payload)
    rows.append(
        {
            "method": FUSION_PROPOSED,
            "family": "proposed",
            "reference": None,
            "primary": dict.fromkeys(PRIMARY_COLUMNS, PENDING),
            "constraint": dict.fromkeys(CONSTRAINT_COLUMNS, PENDING),
            "executed": False,
        }
    )
    return {
        "caption": (
            "Temporal-holdout rediscovery benchmark: pre-specified head-to-head design "
            "against 37 agent, knowledge-graph and mechanistic-modeling baselines across six "
            "architectural families, plus two fusion-mechanism controls."
        ),
        "registry": registry,
        "panels": {"A": list(PRIMARY_COLUMNS), "B": list(CONSTRAINT_COLUMNS)},
        "rows": rows,
    }


TABLE_TWO_METHODS: tuple[tuple[str, str, str, str], ...] = (
    (
        "mrTRG (MRI tumor-regression grade 1-2)",
        "sensitivity / specificity",
        "retrospective / prospective",
        PENDING,
    ),
    (
        "SOC-only nomogram (cT/cN/CEA/MRF), in-cohort",
        "AUROC",
        "retrospective / prospective",
        PENDING,
    ),
    ("Clinical dynamic nomogram", "C-index", "internal / external", PENDING),
    ("NLR/PLR-augmented clinical nomogram", "AUC", "single cohort", PENDING),
    ("PET/CT semiquantitative nomogram", "AUC", "train / validation", PENDING),
    ("Clinical-features-only model (Cmodel)", "AUC", "validation", PENDING),
    ("Clinical-features-only machine-learning model", "AUC", "preliminary / external", PENDING),
    ("CT-radiomics-only model (CTmodel)", "AUC", "validation", PENDING),
    ("MRI-T1-weighted radiomics-only model (T1model)", "AUC", "validation", PENDING),
    ("MRI-T2-weighted radiomics-only model (T2model)", "AUC", "validation", PENDING),
    (
        "MRI radiomics, pooled multi-study meta-analytic estimate",
        "pooled AUROC",
        "meta-analytic",
        PENDING,
    ),
    (
        "Habitat-radiomics tumor-regression-grade model",
        "AUC",
        "imaging-biomarker analysis",
        PENDING,
    ),
    ("MRI radiomics plus ResNet50 deep-learning features", "AUC", "train / external", PENDING),
    ("Clinical deep-learning-radiomics combined model (CDLR)", "AUC", "train / external", PENDING),
    ("Sub-regional radiomics plus deep learning (SRADL)", "AUC", "train / test1 / test2", PENDING),
    ("Integrated multi-omics model (Fmodel)", "AUC", "train / val / internal / external", PENDING),
    ("Long non-coding RNA signature", "AUC", "validation", PENDING),
    ("186-gene transcriptomic signature", "AUC", "cross-validation", PENDING),
    ("Gene-expression pretreatment-biopsy classifier", "AUC", "training / validation", PENDING),
    (
        "CRB-Concordance-informed clinical model (proposed)",
        "AUROC",
        "retrospective / prospective",
        PENDING,
    ),
)


def table_two_snapshot() -> dict[str, object]:
    """Table 2 with every value cell pending, as its caption declares."""

    return {
        "caption": (
            "Clinical-signature and standard-of-care comparators: pre-specified discrimination "
            "benchmark for the downstream clinical association of the top-ranked concordant "
            "candidate's transcriptomic signature, once the clinical arms are analyzed."
        ),
        "rows": [
            {"method": method, "metric": metric, "cohort": cohort, "value": value}
            for method, metric, cohort, value in TABLE_TWO_METHODS
        ],
        "executed": False,
    }


def verdict_counts(verdicts: dict[str, str]) -> dict[str, int]:
    tally = {verdict.value: 0 for verdict in Verdict}
    for value in verdicts.values():
        tally[value] = tally.get(value, 0) + 1
    return tally


def summarise_seed_runs(runs: tuple[dict[str, object], ...]) -> dict[str, float]:
    """Pooled seed-level summary of the executed simulation runs."""

    if not runs:
        return {}
    recalls: list[float] = [float(run["recall_at_k"]) for run in runs]
    conflicts: list[float] = [float(run["mean_conflict"]) for run in runs]
    widths: list[float] = [float(run["mean_interval_width"]) for run in runs]
    briers: list[float] = [float(run["brier"]) for run in runs]
    separabilities: list[float] = [float(run["width_separability"]) for run in runs]
    return {
        "runs": float(len(recalls)),
        "mean_recall": float(np.mean(np.asarray(recalls, dtype=float))),
        "mean_conflict": float(np.mean(np.asarray(conflicts, dtype=float))),
        "mean_interval_width": float(np.mean(np.asarray(widths, dtype=float))),
        "mean_brier": float(np.mean(np.asarray(briers, dtype=float))),
        "mean_width_separability": float(np.mean(np.asarray(separabilities, dtype=float))),
    }


def render_text(title: str, sections: dict[str, object]) -> str:
    """Render a report as plain text, never markdown."""

    lines = [title, "=" * len(title), ""]
    for name, payload in sections.items():
        lines.append(name)
        lines.append("-" * len(name))
        lines.extend(_render_value(payload, indent=0))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_value(payload: object, *, indent: int) -> list[str]:
    pad = " " * indent
    if isinstance(payload, dict):
        lines: list[str] = []
        for key in sorted(payload, key=str):
            value = payload[key]
            if isinstance(value, (dict, list, tuple)):
                lines.append(f"{pad}{key}:")
                lines.extend(_render_value(value, indent=indent + 2))
            else:
                lines.append(f"{pad}{key} = {value}")
        return lines
    if isinstance(payload, (list, tuple)):
        lines = []
        for position, item in enumerate(payload):
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{pad}[{position}]")
                lines.extend(_render_value(item, indent=indent + 2))
            else:
                lines.append(f"{pad}- {item}")
        return lines
    return [f"{pad}{payload}"]


def write_text_report(path: str | Path, title: str, sections: dict[str, object]) -> Path:
    return atomic_write_text(path, render_text(title, sections))
