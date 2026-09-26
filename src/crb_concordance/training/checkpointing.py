"""Atomic checkpointing that carries the seed and the configuration.

Ref: Sec. 4.6 (every reported quantity is reproducible from the archived code, so a
resumed run must restore the seed and the settings it was started with).
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crb_concordance.utils.atomic import _atomic_write

CHECKPOINT_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, malformed or version-incompatible."""


@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    """The non-learned record carried alongside the parameters."""

    version: int
    seed: int
    step: int
    config: dict[str, object]
    digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "step": self.step,
            "config": self.config,
            "digest": self.digest,
        }


def parameter_digest(parameters: Mapping[str, np.ndarray]) -> str:
    """Content digest over tensor payloads, not over container bytes."""

    digest = hashlib.sha256()
    for name in sorted(parameters):
        array = np.ascontiguousarray(np.asarray(parameters[name], dtype=np.float64))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    parameters: Mapping[str, np.ndarray],
    *,
    seed: int,
    step: int,
    config: dict[str, object],
) -> CheckpointMeta:
    """Write parameters and metadata to one file through a temporary and a replace."""

    payload: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "seed": int(seed),
        "step": int(step),
        "config": config,
        "parameters": {name: np.asarray(value) for name, value in parameters.items()},
    }
    buffer = io.BytesIO()
    np.savez(buffer, **{f"param__{name}": value for name, value in payload["parameters"].items()})  # type: ignore[union-attr]
    meta = {
        "version": CHECKPOINT_VERSION,
        "seed": int(seed),
        "step": int(step),
        "config": config,
        "digest": parameter_digest(parameters),
    }
    buffer.write(b"\n__META__" + json.dumps(meta, sort_keys=True).encode("utf-8"))
    target = Path(path)
    _atomic_write(target, buffer.getvalue())
    return CheckpointMeta(
        version=CHECKPOINT_VERSION,
        seed=int(seed),
        step=int(step),
        config=dict(config),
        digest=meta["digest"],  # type: ignore[arg-type]
    )


def load_checkpoint(path: str | Path) -> tuple[dict[str, np.ndarray], CheckpointMeta]:
    target = Path(path)
    if not target.exists():
        raise CheckpointError(f"checkpoint not found: {target}")
    raw = target.read_bytes()
    marker = raw.find(b"\n__META__")
    if marker < 0:
        raise CheckpointError(f"checkpoint carries no metadata block: {target}")
    payload = np.load(io.BytesIO(raw[:marker]), allow_pickle=False)
    parameters = {
        key[len("param__") :]: payload[key] for key in payload.files if key.startswith("param__")
    }
    meta = json.loads(raw[marker + len(b"\n__META__") :].decode("utf-8"))
    if int(meta["version"]) != CHECKPOINT_VERSION:
        raise CheckpointError(f"checkpoint version {meta['version']} is not {CHECKPOINT_VERSION}")
    recomputed = parameter_digest(parameters)
    if recomputed != meta["digest"]:
        raise CheckpointError("checkpoint digest does not match its parameter payload")
    return parameters, CheckpointMeta(
        version=int(meta["version"]),
        seed=int(meta["seed"]),
        step=int(meta["step"]),
        config=dict(meta["config"]),
        digest=str(meta["digest"]),
    )


def resume_seed(path: str | Path) -> int:
    _, meta = load_checkpoint(path)
    return meta.seed
