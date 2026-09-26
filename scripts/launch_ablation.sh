#!/usr/bin/env bash
# The four extraction levels plus the substitution tier.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m crb_concordance.cli.ablate --experiment ablation_substitution_dempster --output-root runs
