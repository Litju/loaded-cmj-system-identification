"""Deterministic observation preprocessing and comparison-grid utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import plant


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_PATH = REPOSITORY_ROOT / "configs" / "preprocessing.json"


def preprocessing_config() -> dict[str, Any]:
    return json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))


def comparison_grid() -> np.ndarray:
    """Return the source comparison grid as a floating-point array in seconds."""

    config = preprocessing_config()
    grid = config.get("comparison_grid", {})
    start = float(grid.get("start_s", grid.get("start_time_s", 0.0)))
    stop = float(grid.get("stop_s", grid.get("end_time_s", 3.6)))
    step = float(grid.get("step_s", 0.01))
    count = int(round((stop - start) / step)) + 1
    return start + step * np.arange(count, dtype=float)


def preprocess_observations(observations: dict[str, Any]) -> dict[str, Any]:
    """Apply the source weighing and net-force transform to observed channels."""

    trace = {key: list(value) if isinstance(value, list) else value for key, value in observations.items()}
    if "bar_velocity_m_s" not in trace:
        time_s = np.asarray(trace["time_s"], dtype=float)
        bar = np.asarray(trace["bar_displacement_m"], dtype=float)
        trace["bar_velocity_m_s"] = np.gradient(bar, time_s).tolist()
    return plant.preprocess_force_trace(trace)


def resample(values: Any, source_time_s: Any, target_time_s: Any) -> np.ndarray:
    """Linearly resample one finite measurement channel in its source units."""

    source_time = np.asarray(source_time_s, dtype=float)
    source_values = np.asarray(values, dtype=float)
    target_time = np.asarray(target_time_s, dtype=float)
    if source_time.ndim != 1 or source_values.ndim != 1 or len(source_time) != len(source_values):
        raise ValueError("source time and values must be equal-length vectors")
    return np.interp(target_time, source_time, source_values)
