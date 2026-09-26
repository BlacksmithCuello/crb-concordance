"""Shared command-line arguments.

Ref: Sec. 4.6 (every reported quantity is reproducible from the archived code, so a
run is fully described by its configuration file plus its overrides).
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CONFIG_ROOT = "configs"


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-root", default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--experiment", default="main")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="dotted key=value override applied on top of the experiment file",
    )
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--quiet", action="store_true")
    return parser


def add_sizing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-auroc", type=float, default=0.70)
    parser.add_argument("--margin", type=float, default=0.081)
    parser.add_argument("--prevalence", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)


def experiment_output(root: str | Path, name: str) -> Path:
    return Path(root) / name
