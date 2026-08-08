"""Canonical parameter authority and vector conversion helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from . import plant


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPOSITORY_ROOT / "configs"


def parameter_schema() -> dict[str, Any]:
    """Return the checked-in schema defining names, units, and bounds."""

    return json.loads((CONFIG_DIR / "param_schema.json").read_text(encoding="utf-8"))


def load_parameters(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a parameter configuration."""

    return plant.load_params(path)


def default_parameters() -> dict[str, Any]:
    """Return the public nominal parameter vector."""

    return plant.default_params()


def load_named_parameters(name: str) -> dict[str, Any]:
    """Load ``configs/<name>.json`` through the same plant validator."""

    return load_parameters(CONFIG_DIR / f"{name}.json")


def _coordinate_spec(name: str, spec: dict[str, Any], index: int | None = None) -> tuple[float, float]:
    if spec.get("type") == "array":
        if index is None:
            raise ValueError(f"array parameter requires an index: {name}")
        item = spec.get("prefixItems", [])[index]
    else:
        item = spec
    if "minimum" not in item or "maximum" not in item:
        raise ValueError(f"parameter schema has no finite bounds for {name}")
    return float(item["minimum"]), float(item["maximum"])


def flatten_parameters(
    params: dict[str, Any],
    names: Iterable[str] | None = None,
) -> tuple[list[str], list[float], list[float], list[float]]:
    """Flatten the full structured parameter vector in schema order.

    Array coordinates are named with an explicit zero-based index, preserving
    the source vector ordering while making optimizer diagnostics unambiguous.
    The return values are ``(names, values, lower_bounds, upper_bounds)``.
    """

    plant.validate_params(params)
    schema = parameter_schema()
    requested = set(names) if names is not None else None
    flat_names: list[str] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for key in schema.get("required", list(schema["properties"])):
        spec = schema["properties"][key]
        value = params[key]
        if spec.get("type") == "array":
            for index, element in enumerate(value):
                coordinate = f"{key}[{index}]"
                if requested is not None and coordinate not in requested:
                    continue
                lo, hi = _coordinate_spec(coordinate, spec, index)
                flat_names.append(coordinate)
                values.append(float(element))
                lower.append(lo)
                upper.append(hi)
        else:
            if requested is not None and key not in requested:
                continue
            lo, hi = _coordinate_spec(key, spec)
            flat_names.append(key)
            values.append(float(value))
            lower.append(lo)
            upper.append(hi)
    if requested is not None and set(flat_names) != requested:
        missing = sorted(requested - set(flat_names))
        if missing:
            raise ValueError(f"unknown parameter coordinates: {missing}")
    return flat_names, values, lower, upper


def unflatten_parameters(
    base: dict[str, Any],
    names: Iterable[str],
    values: Iterable[float],
) -> dict[str, Any]:
    """Return a validated structured vector with selected coordinates replaced."""

    result = copy.deepcopy(base)
    schema = parameter_schema()["properties"]
    for name, value in zip(names, values):
        if name.endswith("]") and "[" in name:
            key, index_text = name.rsplit("[", 1)
            index = int(index_text[:-1])
            if key not in result or not isinstance(result[key], list):
                raise ValueError(f"not an array parameter coordinate: {name}")
            result[key][index] = float(value)
        else:
            if name not in result:
                raise ValueError(f"unknown parameter coordinate: {name}")
            result[name] = int(round(value)) if schema[name].get("type") == "integer" else float(value)
    return plant.validate_params(result)
