"""Readers for the declared public resources.

Ref: Data availability (TCGA-COAD and TCGA-READ through the GDC open tier and the
cBioPortal PanCancer Atlas study; the five pretreatment-biopsy series GSE35452,
GSE45404, GSE68204, GSE119409 and GSE150082; DepMap Public 26Q1 CRISPR Chronos
effect scores with the CCLE metabolomics table; GDSC Release 8.4 and PRISM for drug
response; LINCS L1000 CMap 2020 for transcriptional response).

The readers are schema-based: they parse released files when they are present
under the data root and raise ``SourceUnavailable`` otherwise, so a missing
resource degrades to a reported absence rather than to a fabricated value.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from crb_concordance.cohorts.schema import DependencyObservation

GDC_PROJECT_IDS: tuple[str, ...] = ("TCGA-COAD", "TCGA-READ")
GDC_DATA_RELEASE = "GDC Data Release 46"
GEO_SERIES: tuple[str, ...] = ("GSE35452", "GSE45404", "GSE68204", "GSE119409", "GSE150082")
DEPMAP_RELEASE = "DepMap Public 26Q1"
DEPMAP_LICENSE = "CC-BY 4.0, DepMap terms"
GDSC_RELEASE = "GDSC Release 8.4"
CMAP_RELEASE = "CMap 2020 Level 5"


class SourceUnavailable(FileNotFoundError):
    """Raised when a declared resource is absent from the data root."""


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """One declared resource with its version, licence and expected file name."""

    name: str
    version: str
    license: str
    url: str
    relative_path: str
    access: str = "open"

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "url": self.url,
            "relative_path": self.relative_path,
            "access": self.access,
        }


RESOURCES: tuple[ResourceSpec, ...] = (
    ResourceSpec(
        name="GDC clinical and expression export",
        version=GDC_DATA_RELEASE,
        license="GDC open tier; raw reads under dbGaP phs000178",
        url="https://portal.gdc.cancer.gov",
        relative_path="gdc/gdc_clinical_export.tsv",
    ),
    ResourceSpec(
        name="cBioPortal PanCancer Atlas study",
        version="coadread_tcga_pan_can_atlas_2018",
        license="cBioPortal public-instance terms",
        url="https://www.cbioportal.org/study/summary?id=coadread_tcga_pan_can_atlas_2018",
        relative_path="cbioportal/coadread_tcga_pan_can_atlas_2018",
    ),
    ResourceSpec(
        name="GEO pretreatment-biopsy series",
        version="five series",
        license="GEO public-deposit terms",
        url="https://www.ncbi.nlm.nih.gov/geo/",
        relative_path="geo",
    ),
    ResourceSpec(
        name="DepMap CRISPR gene effect",
        version=DEPMAP_RELEASE,
        license=DEPMAP_LICENSE,
        url="https://depmap.org/portal/data_page/?tab=allData",
        relative_path="depmap/CRISPRGeneEffect.csv",
    ),
    ResourceSpec(
        name="GDSC drug sensitivity",
        version=GDSC_RELEASE,
        license="CC-BY 4.0, Sanger standard terms",
        url="https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/",
        relative_path="gdsc/gdsc_release_8.4.csv",
    ),
    ResourceSpec(
        name="PRISM Repositioning",
        version="DepMap-hosted release",
        license="CC-BY 4.0",
        url="https://depmap.org/repurposing/",
        relative_path="depmap/PRISM_Repositioning.csv",
    ),
    ResourceSpec(
        name="CCLE metabolomics",
        version="Li et al., Nature Medicine 2019",
        license="CC-BY 4.0, DepMap/Broad terms",
        url="https://depmap.org/portal/data_page/?tab=allData",
        relative_path="depmap/CCLE_metabolomics.csv",
    ),
    ResourceSpec(
        name="LINCS L1000 Connectivity Map",
        version=CMAP_RELEASE,
        license="CC0",
        url="https://clue.io/data/CMap2020",
        relative_path="cmap/cmap2020_level5",
    ),
    ResourceSpec(
        name="OptimusKG substrate",
        version="Harvard Dataverse doi:10.7910/DVN/IYNGEV",
        license="MIT for code, CC BY-NC-SA 4.0 for data",
        url="https://github.com/mims-harvard/OptimusKG",
        relative_path="optimuskg",
    ),
    ResourceSpec(
        name="Human-GEM",
        version="v2.0.1",
        license="CC-BY 4.0",
        url="https://github.com/SysBioChalmers/Human-GEM/releases/tag/v2.0.1",
        relative_path="human_gem/Human-GEM.json",
    ),
    ResourceSpec(
        name="Hetionet",
        version="v1.0",
        license="CC0 1.0",
        url="https://github.com/hetio/hetionet",
        relative_path="hetionet",
    ),
    ResourceSpec(
        name="Reactome pathway data",
        version="current download",
        license="CC0",
        url="https://reactome.org/download-data",
        relative_path="reactome",
    ),
    ResourceSpec(
        name="KEGG colorectal cancer pathway map",
        version="current",
        license="academic use; no commercial reuse terms",
        url="https://www.genome.jp/kegg/pathway.html",
        relative_path="kegg/",
    ),
    ResourceSpec(
        name="MetaboLights open metabolomics route",
        version="current",
        license="EMBL-EBI terms of use",
        url="https://www.ebi.ac.uk/metabolights/",
        relative_path="metabolights",
    ),
)

PRIVATE_RESOURCES: tuple[str, ...] = (
    "prospective arm, three sites",
    "retrospective arm, three sites",
    "in vitro cell-line corroboration",
)


@dataclass(frozen=True, slots=True)
class ResourceStatus:
    spec: ResourceSpec
    present: bool

    def as_dict(self) -> dict[str, object]:
        return {**self.spec.as_dict(), "present": self.present}


def inventory(data_root: str | Path) -> tuple[ResourceStatus, ...]:
    root = Path(data_root)
    return tuple(
        ResourceStatus(spec=spec, present=(root / spec.relative_path).exists())
        for spec in RESOURCES
    )


def _require(path: Path, spec_name: str) -> Path:
    if not path.exists():
        raise SourceUnavailable(f"{spec_name} is not present at {path}")
    return path


def read_gdc_clinical(path: str | Path) -> tuple[dict[str, str], ...]:
    """Read a GDC clinical export with ``case_id`` and ``project_id`` columns."""

    target = _require(Path(path), "GDC clinical export")
    with target.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or "case_id" not in reader.fieldnames:
            raise SourceUnavailable(f"GDC export is missing the case_id column: {target}")
        rows = tuple(dict(row) for row in reader)
    for row in rows:
        if row.get("project_id") not in GDC_PROJECT_IDS:
            raise SourceUnavailable(f"unexpected project in GDC export: {row.get('project_id')!r}")
    return rows


def read_geo_series_matrix(path: str | Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read the sample identifiers and gene symbols of a GEO series matrix."""

    target = _require(Path(path), "GEO series matrix")
    samples: tuple[str, ...] = ()
    genes: list[str] = []
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("!Sample_geo_accession"):
            samples = tuple(part.strip('"') for part in line.split("\t")[1:])
        elif line and not line.startswith("!") and "\t" in line:
            genes.append(line.split("\t", 1)[0].strip('"'))
    if not samples:
        raise SourceUnavailable(f"no sample accessions found in {target}")
    return samples, tuple(genes)


def read_depmap_gene_effect(path: str | Path) -> tuple[DependencyObservation, ...]:
    """Read a DepMap CRISPR gene-effect matrix, one column per gene."""

    target = _require(Path(path), "DepMap gene effect")
    with target.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if not header or not header[0]:
            raise SourceUnavailable(f"DepMap gene-effect matrix has no header: {target}")
        symbols = [column.split(" (")[0] for column in header[1:]]
        collected: list[DependencyObservation] = []
        for row in reader:
            if not row:
                continue
            cell_line = row[0]
            for symbol, value in zip(symbols, row[1:], strict=False):
                if value in ("", "NA"):
                    continue
                collected.append(
                    DependencyObservation(
                        cell_line=cell_line,
                        gene=symbol,
                        chronos_effect=float(value),
                        drug_response=None,
                        tissue="",
                        primary_disease="colorectal adenocarcinoma",
                    )
                )
    return tuple(collected)


def read_drug_response(path: str | Path, *, gene_column: str = "gene") -> dict[str, float]:
    """Read a single-gene drug-response export keyed by gene symbol."""

    target = _require(Path(path), "drug response table")
    with target.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or gene_column not in reader.fieldnames:
            raise SourceUnavailable(f"drug-response table lacks the {gene_column} column: {target}")
        payload: dict[str, float] = {}
        for row in reader:
            value = row.get("response")
            if value in (None, "", "NA"):
                continue
            payload[row[gene_column]] = float(value)
    return payload


def read_optimuskg_edges(path: str | Path) -> tuple[dict[str, object], ...]:
    """Read the substrate edge table from a Dataverse JSON export."""

    target = _require(Path(path), "OptimusKG edge table")
    payload = json.loads(target.read_text(encoding="utf-8"))
    records = payload.get("edges")
    if not isinstance(records, list):
        raise SourceUnavailable(f"OptimusKG export has no edge list: {target}")
    return tuple(dict(record) for record in records)
