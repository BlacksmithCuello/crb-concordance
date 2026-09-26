#!/usr/bin/env bash
# Discount rates and the evidence-to-mass map.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m crb_concordance.cli.calibrate --experiment main --output-root runs
python -m crb_concordance.cli.discover --experiment main --output-root runs
