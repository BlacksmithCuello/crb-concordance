"""Configuration loading with dotted command-line overrides.

Ref: Sec. 4.6 (every reported quantity is reproducible from the archived code).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCALARS: tuple[type, ...] = (int, float, bool)


class ConfigError(ValueError):
    """Raised when a configuration file or override cannot be interpreted."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"configuration file not found: {target}")
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration root must be a mapping: {target}")
    return payload


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def coerce_scalar(text: str) -> Any:
    lowered = text.strip()
    if lowered.lower() in {"true", "false"}:
        return lowered.lower() == "true"
    if lowered.lower() in {"null", "none"}:
        return None
    try:
        return int(lowered)
    except ValueError:
        pass
    try:
        return float(lowered)
    except ValueError:
        pass
    if lowered.startswith("[") and lowered.endswith("]"):
        inner = lowered[1:-1].strip()
        if not inner:
            return []
        return [coerce_scalar(part.strip()) for part in inner.split(",")]
    return text


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must be key=value: {item}")
        dotted, raw = item.split("=", 1)
        cursor: dict[str, Any] = result
        keys = dotted.split(".")
        for key in keys[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        cursor[keys[-1]] = coerce_scalar(raw)
    return result


def _rebase(overrides: list[str], block: str) -> list[str]:
    """Strip a block prefix so the override lands inside that block."""

    prefix = f"{block}."
    return [item[len(prefix) :] for item in overrides if item.startswith(prefix)]


@dataclass(slots=True)
class ExperimentConfig:
    """Flattened view of one experiment: data, model, train and experiment blocks."""

    name: str
    experiment: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    overrides: list[str] = field(default_factory=list)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a dotted key; an unprefixed key resolves against the experiment block."""

        keys = dotted.split(".")
        if keys[0] in ("data", "model", "train"):
            cursor: Any = {"data": self.data, "model": self.model, "train": self.train}[keys[0]]
            keys = keys[1:]
        else:
            cursor = self.experiment
        for key in keys:
            if not isinstance(cursor, dict) or key not in cursor:
                return default
            cursor = cursor[key]
        return cursor

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required key: {dotted}")
        return value


def load_experiment(
    config_root: str | Path,
    experiment: str,
    *,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    root = Path(config_root)
    blocks: dict[str, dict[str, Any]] = {}
    experiment_payload = load_yaml(root / "experiment" / f"{experiment}.yaml")
    for block in ("data", "model", "train"):
        group = experiment_payload.get(block)
        if isinstance(group, dict):
            blocks[block] = dict(group)
        elif isinstance(group, str):
            blocks[block] = load_yaml(root / block / f"{group}.yaml")
        else:
            blocks[block] = {}
    for block in ("data", "model", "train"):
        inline = experiment_payload.get(f"{block}_override")
        if isinstance(inline, dict):
            blocks[block] = deep_merge(blocks[block], inline)
    applied = overrides or []
    experiment_block = {
        key: value
        for key, value in experiment_payload.items()
        if key not in ("data", "model", "train")
    }
    experiment_block = apply_overrides(
        experiment_block,
        [item for item in applied if not item.startswith(("data.", "model.", "train."))],
    )
    return ExperimentConfig(
        name=experiment,
        experiment=experiment_block,
        data=apply_overrides(blocks["data"], _rebase(applied, "data")),
        model=apply_overrides(blocks["model"], _rebase(applied, "model")),
        train=apply_overrides(blocks["train"], _rebase(applied, "train")),
        overrides=[
            o for o in applied if not o.startswith(("data.", "model.", "train.", "experiment."))
        ],
    )
