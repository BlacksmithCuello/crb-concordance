#!/usr/bin/env bash
# The in silico simulation study of Supplementary Table S1.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m crb_concordance.cli.simulate --experiment main --output-root runs
