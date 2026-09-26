#!/usr/bin/env bash
# The temporal-holdout rediscovery harness on the substrate pool.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python -m crb_concordance.cli.benchmark --experiment supplementary_edge_holdout --output-root runs
