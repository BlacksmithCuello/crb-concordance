"""Atomic file writes and content digests.

Ref: Sec. 4.2 (the ledger is append-only and must never be observed half written).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

DIGEST_CHUNK = 1 << 20
SKIP_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".verify",
}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def _atomic_write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp creates 0600; the release artefacts must stay world readable.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return _atomic_write(Path(path), text.encode("utf-8"))


def atomic_write_json(path: str | Path, payload: Any, *, sort_keys: bool = True) -> Path:
    text = json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=False)
    return atomic_write_text(path, text + "\n")


def append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    block = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records)
    return atomic_write_text(target, existing + block)


def sha256_file(path: str | Path, *, chunk: int = DIGEST_CHUNK) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def iter_tree_files(root: str | Path) -> list[Path]:
    base = Path(root)
    found: list[Path] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if SKIP_DIRECTORIES.intersection(path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        found.append(path)
    return found


def tree_digest(root: str | Path) -> str:
    digest = hashlib.sha256()
    base = Path(root)
    for path in iter_tree_files(base):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
