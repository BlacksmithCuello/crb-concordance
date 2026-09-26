#!/usr/bin/env bash
# Point the readers at the declared public resources.
#
# The release bundles no third-party data. Download each resource from the URL in
# dataset_urls.txt into data/ using the relative path that
# crb_concordance.cohorts.public_cohorts declares, and the corresponding reader will
# pick it up. A resource that is absent raises SourceUnavailable, which is reported
# as NOT_RUN rather than producing a value.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/gdc data/geo data/depmap data/gdsc data/cmap data/optimuskg data/human_gem data/hetionet data/reactome data/kegg data/metabolights
echo "data root ready: $(pwd)/data"
echo "declare each resource by following dataset_urls.txt"
