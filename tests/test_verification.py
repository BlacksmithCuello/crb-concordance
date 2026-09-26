"""The verification layer: the artefact writers and the shipped artefacts."""

from __future__ import annotations

import json

import numpy as np

from crb_concordance.cli.verify import (
    CLAIM_NAME,
    CLAIMS,
    DEVIATIONS,
    MANIFEST_NAME,
    NOT_REPRODUCED,
    REPORT_NAME,
    SUMMARY_NAME,
    ClaimSpec,
    build_claim_payload,
    claim_records,
    resolve_root,
    write_integrity_manifest,
)
from crb_concordance.evaluation.audit import CheckRecorder, status_counts
from crb_concordance.utils.atomic import iter_tree_files, sha256_file
from crb_concordance.utils.types import Verdict


def test_recorder_counts_every_verdict() -> None:
    recorder = CheckRecorder()
    recorder.ok("a", "g", "fine")
    recorder.bad("b", "g", "broken")
    recorder.not_run("c", "g", "no data")
    recorder.blocked("d", "g", "gated")
    counts = status_counts(recorder.results)
    assert counts["PASS"] == 1
    assert counts["FAIL"] == 1
    assert counts["NOT_RUN"] == 1
    assert counts["BLOCKED"] == 1


def test_claim_spec_reports_a_missing_symbol(tmp_path) -> None:
    module = tmp_path / "src" / "pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text("def present() -> None:\n    return None\n", encoding="utf-8")
    good = ClaimSpec("C", "Sec. 1", "statement", ("src/pkg/mod.py",), ("def present",))
    bad = ClaimSpec("C2", "Sec. 1", "statement", ("src/pkg/mod.py",), ("def absent",))
    assert good.as_record(tmp_path)["verification"] == "PASS"
    record = bad.as_record(tmp_path)
    assert record["verification"] == "FAIL"
    assert "unresolved" in record["statement"]


def test_claim_records_cover_the_declared_artefacts(release_root) -> None:
    records = claim_records(release_root, CLAIMS)
    assert len(records) == len(CLAIMS) >= 40
    assert all(record["code_paths"] for record in records)
    assert all(record["paper_location"] for record in records)


def test_every_located_claim_resolves_on_this_tree(release_root) -> None:
    records = claim_records(release_root, CLAIMS)
    unresolved = [record["claim_id"] for record in records if record["verification"] != "PASS"]
    assert unresolved == [], f"unresolved claims: {unresolved}"


def test_deviations_and_unreproduced_entries_are_justified() -> None:
    assert len(DEVIATIONS) >= 8
    assert all(entry["paper_location"] and entry["justification"] for entry in DEVIATIONS)
    assert all(entry["item"] and entry["reason"] for entry in NOT_REPRODUCED)


def test_claim_payload_is_self_consistent(release_root) -> None:
    payload = build_claim_payload(release_root, CLAIMS, {})
    assert payload["overall"] in {"PASS", "PARTIALLY_VERIFIED"}
    assert payload["unresolved_claims"] == []
    assert payload["paper"].startswith("Knowledge-Graph-Augmented")


def test_manifest_writer_is_parameterised_by_root(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("first\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("value = 1\n", encoding="utf-8")
    payload = write_integrity_manifest(tmp_path)
    paths = {entry["path"] for entry in payload["files"]}
    assert paths == {"a.txt", "src/b.py"}
    assert payload["file_count"] == 2
    assert payload["algorithm"] == "SHA-256"
    assert MANIFEST_NAME not in paths
    assert (tmp_path / MANIFEST_NAME).exists()
    assert len(payload["manifest_digest"]) == 64


def test_manifest_digest_reflects_content(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("first\n", encoding="utf-8")
    first = write_integrity_manifest(tmp_path)["manifest_digest"]
    (tmp_path / "a.txt").write_text("second\n", encoding="utf-8")
    second = write_integrity_manifest(tmp_path)["manifest_digest"]
    assert first != second


def test_shipped_artefacts_exist_and_agree(release_root) -> None:
    report_path = release_root / REPORT_NAME
    claim_path = release_root / CLAIM_NAME
    summary_path = release_root / SUMMARY_NAME
    manifest_path = release_root / MANIFEST_NAME
    for path in (report_path, claim_path, summary_path, manifest_path):
        assert path.exists(), path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    claims = json.loads(claim_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(report["checks"]) >= 70
    assert report["overall"] in {"PASS", "PARTIALLY_VERIFIED"}
    # A failing check must be a measured finding, not an unexplained defect: every FAIL
    # has to be documented in the release's findings list.
    documented = {finding["check"] for finding in claims["findings"]}
    failing = {check["name"] for check in report["checks"] if check["status"] == "FAIL"}
    assert failing <= documented, f"undocumented failures: {sorted(failing - documented)}"
    assert report["status_counts"]["PASS"] >= 80
    assert claims["unresolved_claims"] == []
    assert len(manifest["files"]) == manifest["file_count"]
    assert claims["overall"] in {"PASS", "PARTIALLY_VERIFIED"}
    # A test that called the writer would have replaced the shipped report with a
    # partial one; the full run must leave every check present.
    names = {check["name"] for check in report["checks"]}
    assert {"mass_simplex_identity", "conflict_monotonicity_scan", "fba_growth_feasible"} <= names
    assert {"conflict_trace_h2", "hanley_mcneil_sizing", "effective_line_count"} <= names


def test_shipped_manifest_covers_every_file_of_the_tree(release_root) -> None:
    manifest = json.loads((release_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    listed = {entry["path"] for entry in manifest["files"]}
    on_disk = {
        path.relative_to(release_root).as_posix()
        for path in iter_tree_files(release_root)
        if path.name != MANIFEST_NAME
    }
    assert listed == on_disk
    for entry in manifest["files"]:
        target = release_root / entry["path"]
        assert sha256_file(target) == entry["sha256"], entry["path"]


def test_shipped_report_matches_its_own_checks(release_root) -> None:
    report = json.loads((release_root / REPORT_NAME).read_text(encoding="utf-8"))
    counted = {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}
    for check in report["checks"]:
        counted[check["status"]] += 1
    assert counted == report["status_counts"]
    assert report["outstanding"] == [
        check["name"]
        for check in report["checks"]
        if check["status"] in {"FAIL", "NOT_RUN", "BLOCKED"}
    ]


def test_shipped_summary_lists_the_outstanding_checks(release_root) -> None:
    text = (release_root / SUMMARY_NAME).read_text(encoding="utf-8")
    report = json.loads((release_root / REPORT_NAME).read_text(encoding="utf-8"))
    assert "Verification summary" in text
    assert report["overall"] in text
    for name in report["outstanding"]:
        assert name in text


def test_resolve_root_points_at_the_release(release_root) -> None:
    resolved = resolve_root(release_root / "configs")
    assert resolved == release_root


def test_privacy_of_the_shipped_reports(release_root) -> None:
    import re

    pattern = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")
    for name in (REPORT_NAME, CLAIM_NAME, SUMMARY_NAME, "dataset_urls.txt"):
        text = (release_root / name).read_text(encoding="utf-8")
        assert not pattern.search(text), name


def test_check_status_enum_is_exhaustive() -> None:
    assert {verdict.value for verdict in Verdict} == {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}


def test_claim_rows_use_readable_scalars() -> None:
    values = np.asarray([0.1, 0.2])
    assert json.dumps(values.tolist()) == "[0.1, 0.2]"


def test_release_tree_has_no_stray_markdown(release_root) -> None:
    found = sorted(
        path.relative_to(release_root).as_posix()
        for path in iter_tree_files(release_root)
        if path.suffix == ".md"
    )
    assert found == ["README.md"]
