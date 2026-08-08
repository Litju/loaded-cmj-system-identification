"""Read-only source/target compiled-model and rollout equivalence audit.

The source checkout is supplied at runtime and is never copied into this
repository. A numerical comparison is also made when a worktree rollout is
rejected by its native stability contract, but that native status is reported
separately and is never treated as a successful scientific trial.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _descriptor(trial: dict[str, Any]) -> dict[str, Any]:
    descriptor = {
        key: trial[key]
        for key in (
            "depth_scale",
            "braking_duration_scale",
            "propulsion_duration_scale",
        )
        if key in trial
    }
    descriptor["external_load_kg"] = float(trial.get("bar_load_kg", 20.0))
    return descriptor


def _capture(plant: Any, params: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    contact_count: list[int] = []
    original_step = mujoco.mj_step

    def hook(model: Any, data: Any, *args: Any, **kwargs: Any) -> None:
        original_step(model, data, *args, **kwargs)
        qpos.append(np.asarray(data.qpos, dtype=float).copy())
        qvel.append(np.asarray(data.qvel, dtype=float).copy())
        contact_count.append(int(data.ncon))

    original_validator = plant._validate_rollout_result
    plant._validate_rollout_result = lambda *args, **kwargs: None
    mujoco.mj_step = hook
    try:
        result = plant.run_trial(params, trial, record=True)
    finally:
        mujoco.mj_step = original_step
        plant._validate_rollout_result = original_validator
    return {
        "result": result,
        "qpos": np.asarray(qpos, dtype=float),
        "qvel": np.asarray(qvel, dtype=float),
        "contact_count": np.asarray(contact_count, dtype=int),
    }


def _native_rollout_status(
    plant: Any, params: dict[str, Any], trial: dict[str, Any]
) -> dict[str, Any]:
    """Run the plant with its own validator and retain any rejection reason."""

    try:
        result = plant.run_trial(params, trial, record=True)
    except Exception as exc:  # the report must preserve the exact native failure
        return {
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "valid": bool(result.get("valid", False)),
        "error": None,
    }


def _max_diff(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _event_error(left: Any, right: Any, key: str) -> float:
    a = left.get(key)
    b = right.get(key)
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(float(a) - float(b))


def run_equivalence(source_root: str | Path, trial_id: str = "20kg_nominal_a", tolerance: float = 1e-12) -> dict[str, Any]:
    """Compare the target against one trial from a separately located source."""

    source_root = Path(source_root).resolve()
    sys.modules.pop("_loaded_cmj_clean_core", None)
    source_plant = _load_module("loaded_cmj_source_plant", source_root / "data" / "plant.py")
    # Both directly loaded plant modules use a guarded helper-module name;
    # release the source binding before loading the target so each rollout
    # retains its own clean-core implementation.
    sys.modules.pop("_loaded_cmj_clean_core", None)
    target_plant = _load_module("loaded_cmj_target_plant", ROOT / "src" / "loaded_cmj" / "plant.py")
    sys.path.insert(0, str(ROOT / "src"))
    from loaded_cmj.model import compiled_model_signature

    data = json.loads((ROOT / "data" / "identification_trials.json").read_text(encoding="utf-8"))
    trial = next(item for item in data["trials"] if item["trial_id"] == trial_id)
    params = json.loads((ROOT / "configs" / "synthetic_reference.json").read_text(encoding="utf-8"))
    descriptor = _descriptor(trial)

    source_model = source_plant.build_model(params, descriptor)
    target_model = target_plant.build_model(params, descriptor)
    source_signature = compiled_model_signature(source_model)
    target_signature = compiled_model_signature(target_model)
    structural_keys = [
        "nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsite", "nmesh", "ntendon",
        "nsensor", "npair", "neq", "bodies", "joints", "geoms", "sites", "actuators",
        "sensors", "tendons", "timestep_s", "integrator", "iterations", "tolerance", "cone",
    ]
    compiled_equal = all(source_signature[key] == target_signature[key] for key in structural_keys)
    compiled_array_error = max(
        _max_diff(source_signature[key], target_signature[key])
        for key in (
            "body_mass_kg", "body_inertia_kg_m2", "jnt_stiffness", "dof_damping",
            "geom_solref", "pair_solref", "tendon_stiffness", "tendon_damping",
        )
    )
    source_native = _native_rollout_status(source_plant, params, descriptor)
    target_native = _native_rollout_status(target_plant, params, descriptor)
    source_capture = _capture(source_plant, params, descriptor)
    target_capture = _capture(target_plant, params, descriptor)
    source_result = source_capture["result"]
    target_result = target_capture["result"]
    source_traces = source_result["traces"]
    target_traces = target_result["traces"]
    trajectory_error = max(
        _max_diff(source_capture["qpos"], target_capture["qpos"]),
        _max_diff(source_capture["qvel"], target_capture["qvel"]),
    )
    force_error = max(
        _max_diff(source_traces["fz_left_N"], target_traces["fz_left_N"]),
        _max_diff(source_traces["fz_right_N"], target_traces["fz_right_N"]),
        _max_diff(source_traces["fz_total_N"], target_traces["fz_total_N"]),
    )
    lpt_error = max(
        _max_diff(source_traces["bar_displacement_m"], target_traces["bar_displacement_m"]),
        _max_diff(source_traces["bar_velocity_m_s"], target_traces["bar_velocity_m_s"]),
    )
    contact_equal = all(
        np.array_equal(np.asarray(source_traces[key], dtype=bool), np.asarray(target_traces[key], dtype=bool))
        for key in ("left_foot_contact", "right_foot_contact")
    )
    event_error = max(
        _event_error(source_result["events"], target_result["events"], key)
        for key in ("movement_onset_time_s", "takeoff_time_s", "landing_time_s")
    )
    source_valid = bool(source_native["valid"])
    target_valid = bool(target_native["valid"])
    trajectory_equal = trajectory_error <= tolerance
    measurement_equal = max(force_error, lpt_error) <= tolerance
    event_equal = contact_equal and event_error <= tolerance
    return {
        "trial_id": trial_id,
        "tolerance": tolerance,
        "compiled_model_equivalence": "PASS" if compiled_equal and compiled_array_error <= tolerance else "FAIL",
        "compiled_array_max_error": compiled_array_error,
        "source_counts": {key: source_signature[key] for key in structural_keys[:12]},
        "target_counts": {key: target_signature[key] for key in structural_keys[:12]},
        "trajectory_max_error": trajectory_error,
        "force_plate_max_error": force_error,
        "lpt_max_error": lpt_error,
        "event_timing_max_error": event_error,
        "contact_state_equal": contact_equal,
        "source_valid": source_valid,
        "target_valid": target_valid,
        "source_native_validation_error": source_native["error"],
        "target_native_validation_error": target_native["error"],
        "source_diagnostics": source_result.get("diagnostics", {}),
        "target_diagnostics": target_result.get("diagnostics", {}),
        "trajectory_equivalence": "PASS" if trajectory_equal else "FAIL",
        "measurement_equivalence": "PASS" if measurement_equal else "FAIL",
        "event_equivalence": "PASS" if event_equal else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit compiled-model and rollout equivalence.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--trial-id", default="20kg_nominal_a")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_equivalence(args.source_root, args.trial_id, args.tolerance)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if all(report[key] == "PASS" for key in (
        "compiled_model_equivalence", "trajectory_equivalence", "measurement_equivalence", "event_equivalence"
    )) else 1


if __name__ == "__main__":
    raise SystemExit(main())
