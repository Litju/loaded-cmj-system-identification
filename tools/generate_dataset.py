#!/usr/bin/env python3
"""Deterministic loaded-CMJ identification and validation dataset generator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


TASK_ID = "loaded-cmj-system-identification"
SCHEMA_VERSION = "loaded-cmj-dataset-v1"
GENERATION_SEED = 503_117
GENERATOR_NAME = "loaded_cmj_5l_5c_mujoco_primary_dataset_generator"
GENERATOR_VERSION = "1.5.0"
PUBLIC_EXTERNAL_LOAD_KG = 20.0
QUALIFIED_ASYMMETRY_ALPHA = 0.02

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
DATA_DIR = REPOSITORY_ROOT / "data"
CONFIG_DIR = REPOSITORY_ROOT / "configs"
SRC_PLANT_PATH = REPOSITORY_ROOT / "src" / "loaded_cmj" / "plant.py"
PLANT_LOAD_KEY = "external_load_kg"

GRID_START_S = 0.0
GRID_END_S = 3.60
GRID_DT_S = 0.01
GRID_SAMPLE_COUNT = int(round((GRID_END_S - GRID_START_S) / GRID_DT_S)) + 1
COMMON_GRID_S = [round(GRID_START_S + i * GRID_DT_S, 2) for i in range(GRID_SAMPLE_COUNT)]

OBSERVATION_KEYS = [
    "time_s",
    "fz_left_N",
    "fz_right_N",
    "fz_total_N",
    "bar_displacement_m",
    "bar_velocity_m_s",
]
EVENT_KEYS = [
    "weighing_start_time_s",
    "weighing_end_time_s",
    "movement_onset_time_s",
    "takeoff_time_s",
]
SUMMARY_KEYS = [
    "quiet_baseline_mean_N",
    "quiet_baseline_sd_N",
    "takeoff_velocity_m_s",
    "jump_height_im_m",
    "airborne_duration_s",
    "propulsive_impulse_Ns",
    "bar_displacement_range_m",
    "bar_peak_velocity_m_s",
]

SYNTHETIC_REFERENCE_PARAMS = {
    "body_mass_kg": 78.37,
    "joint_stiffness_Nm_rad": [6840.0, 8765.0, 5125.0],
    "joint_damping_Nm_s_rad": [235.0, 282.0, 196.0],
    "braking_gain": 1.67,
    "propulsive_gain": 2.18,
    "activation_delay_s": 0.041,
    "activation_time_constant_s": 0.073,
    "bar_rack_stiffness_N_m": 29550.0,
    "bar_rack_damping_N_s_m": 1035.0,
    "hand_grip_stiffness_N_m": 19150.0,
    "hand_grip_damping_N_s_m": 675.0,
    "contact_stiffness_N_m": 173500.0,
    "contact_damping_N_s_m": 7200.0,
    "force_plate_bias_N": 4.7,
    "force_plate_scale": 0.992,
    "encoder_scale": 1.012,
    "encoder_delay_steps": 2,
    "encoder_filter_tau_s": 0.016,
    "encoder_offset_m": 0.0045,
    "bar_attachment_offset_m": [0.006, -0.008],
    "lpt_tether_stiffness_N_m": 0.0,
    "lpt_tether_damping_N_s_m": 0.0,
}

PUBLIC_TRIAL_SPECS = [
    {
        "trial_id": "20kg_nominal_a",
        "trial_group": "nominal_repeatability",
        "external_load_kg": PUBLIC_EXTERNAL_LOAD_KG,
        "depth_scale": 1.0,
        "braking_duration_scale": 1.0,
        "propulsion_duration_scale": 1.0,
        "drive_asymmetry_alpha": 0.0,
        "noise_seed": 61101,
        "force_noise_sd_N": 1.0,
        "bar_displacement_noise_sd_m": 0.00010,
    },
    {
        "trial_id": "20kg_nominal_b",
        "trial_group": "nominal_repeatability",
        "external_load_kg": PUBLIC_EXTERNAL_LOAD_KG,
        "depth_scale": 1.0,
        "braking_duration_scale": 1.0,
        "propulsion_duration_scale": 1.0,
        "drive_asymmetry_alpha": 0.0,
        "noise_seed": 61102,
        "force_noise_sd_N": 1.0,
        "bar_displacement_noise_sd_m": 0.00010,
    },
    {
        "trial_id": "20kg_depth",
        "trial_group": "countermovement_depth",
        "external_load_kg": PUBLIC_EXTERNAL_LOAD_KG,
        "depth_scale": 1.12,
        "braking_duration_scale": 1.0,
        "propulsion_duration_scale": 1.0,
        "drive_asymmetry_alpha": 0.0,
        "noise_seed": 61103,
        "force_noise_sd_N": 1.0,
        "bar_displacement_noise_sd_m": 0.00010,
    },
    {
        "trial_id": "20kg_timing",
        "trial_group": "phase_timing",
        "external_load_kg": PUBLIC_EXTERNAL_LOAD_KG,
        "depth_scale": 1.0,
        "braking_duration_scale": 1.04,
        "propulsion_duration_scale": 1.02,
        "drive_asymmetry_alpha": 0.0,
        "noise_seed": 61104,
        "force_noise_sd_N": 1.0,
        "bar_displacement_noise_sd_m": 0.00010,
    },
    {
        "trial_id": "20kg_depth_timing",
        "trial_group": "countermovement_depth_timing",
        "external_load_kg": PUBLIC_EXTERNAL_LOAD_KG,
        "depth_scale": 1.12,
        "braking_duration_scale": 0.96,
        "propulsion_duration_scale": 1.02,
        "drive_asymmetry_alpha": 0.0,
        "noise_seed": 61105,
        "force_noise_sd_N": 1.0,
        "bar_displacement_noise_sd_m": 0.00010,
    },
    {
        "trial_id": "20kg_bilateral_asymmetry",
        "trial_group": "synthetic_interlimb_drive_asymmetry",
        "external_load_kg": PUBLIC_EXTERNAL_LOAD_KG,
        "depth_scale": 1.0,
        "braking_duration_scale": 1.0,
        "propulsion_duration_scale": 1.0,
        "drive_asymmetry_alpha": QUALIFIED_ASYMMETRY_ALPHA,
        "noise_seed": 61106,
        "force_noise_sd_N": 1.0,
        "bar_displacement_noise_sd_m": 0.00010,
    },
]


class SplitMix64:
    """Tiny deterministic generator with stable normal samples."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform_open(self) -> float:
        return ((self.next_u64() >> 11) + 0.5) / float(1 << 53)

    def normal(self) -> float:
        u1 = self.uniform_open()
        u2 = self.uniform_open()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def generate(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Generate all public artifacts into ``output_dir`` or the repository.

    Supplying a scratch directory is recommended for regeneration audits. The
    generator uses the same 2 ms MuJoCo plant as the checked-in fixtures.
    """

    root = (Path(output_dir).resolve() if output_dir is not None else REPOSITORY_ROOT)
    data_dir = root / "data"
    config_dir = root / "configs"
    plant = _load_plant()
    _validate_synthetic_reference_params(plant)

    public_trials = [
        _make_trial(plant, spec, split="public", include_validation_descriptor=False)
        for spec in PUBLIC_TRIAL_SPECS
    ]
    validation_trials = [
        _make_trial(plant, spec, split="validation", include_validation_descriptor=True)
        for spec in _validation_trial_specs()
    ]

    public_bundle = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "split": "public",
        "trials": public_trials,
    }
    preprocessing = _preprocessing_contract()
    validation_bundle = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "split": "validation",
        "trials": validation_trials,
    }

    files = {
        data_dir / "identification_trials.json": public_bundle,
        config_dir / "preprocessing.json": preprocessing,
        data_dir / "validation_trials.json": validation_bundle,
        config_dir / "synthetic_reference.json": copy.deepcopy(SYNTHETIC_REFERENCE_PARAMS),
    }
    for path, obj in files.items():
        _write_json(path, obj)

    manifest = _dataset_manifest()
    manifest["generated_file_sha256"] = {
        "data/identification_trials.json": _sha256(data_dir / "identification_trials.json"),
        "configs/preprocessing.json": _sha256(config_dir / "preprocessing.json"),
        "data/validation_trials.json": _sha256(data_dir / "validation_trials.json"),
        "configs/synthetic_reference.json": _sha256(config_dir / "synthetic_reference.json"),
    }
    manifest["public_artifact_hashes"] = {
        "src/loaded_cmj/plant.py": _sha256(SRC_PLANT_PATH),
        "assets/loaded_cmj_model.xml": _sha256(REPOSITORY_ROOT / "assets" / "loaded_cmj_model.xml"),
        "configs/param_schema.json": _sha256(REPOSITORY_ROOT / "configs" / "param_schema.json"),
    }
    _write_json(data_dir / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate loaded-CMJ research datasets.")
    parser.add_argument("--output-dir", type=Path, help="repository or scratch output directory")
    args = parser.parse_args()
    generate(output_dir=args.output_dir)


def _load_plant() -> Any:
    plant_path = SRC_PLANT_PATH
    spec = importlib.util.spec_from_file_location("loaded_cmj_public_plant", plant_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import plant module from {plant_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_synthetic_reference_params(plant: Any) -> None:
    schema_path = CONFIG_DIR / "param_schema.json"
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    if schema.get("type") != "object":
        raise ValueError("param_schema.json must define an object schema")
    plant.validate_params(copy.deepcopy(SYNTHETIC_REFERENCE_PARAMS))


def _make_trial(
    plant: Any,
    spec: dict[str, Any],
    split: str,
    include_validation_descriptor: bool,
) -> dict[str, Any]:
    trial_descriptor = {
        PLANT_LOAD_KEY: spec["external_load_kg"],
        "depth_scale": spec["depth_scale"],
        "braking_duration_scale": spec["braking_duration_scale"],
        "propulsion_duration_scale": spec["propulsion_duration_scale"],
        "drive_asymmetry_alpha": spec.get("drive_asymmetry_alpha", 0.0),
        "duration_s": GRID_END_S,
        "dt_s": plant.DT,
    }
    rollout = plant.run_trial(copy.deepcopy(SYNTHETIC_REFERENCE_PARAMS), trial_descriptor, record=True)
    raw_trace = rollout["traces"]
    observed_trace = _observed_resampled_trace(
        raw_trace,
        noise_seed=int(spec["noise_seed"]),
        force_noise_sd_N=float(spec["force_noise_sd_N"]),
        bar_displacement_noise_sd_m=float(spec["bar_displacement_noise_sd_m"]),
    )
    events = plant.detect_events(observed_trace)
    summary = plant.summarize_trial(observed_trace)

    trial = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "split": split,
        "trial_id": spec["trial_id"],
        "trial_group": spec["trial_group"],
    }
    trial["external_load_kg"] = _round(float(spec["external_load_kg"]), 3)
    trial["bar_load_kg"] = trial["external_load_kg"]
    trial.update({
        "depth_scale": _round(float(spec["depth_scale"]), 4),
        "braking_duration_scale": _round(float(spec["braking_duration_scale"]), 4),
        "propulsion_duration_scale": _round(float(spec["propulsion_duration_scale"]), 4),
        "drive_asymmetry_alpha": _round(float(spec.get("drive_asymmetry_alpha", 0.0)), 4),
    })
    if spec.get("drive_asymmetry_alpha", 0.0) != 0.0:
        trial["known_excitation"] = "synthetic_interlimb_drive_asymmetry"
    if include_validation_descriptor:
        trial.update(_validation_descriptor_fields(spec))
        if "repeat_id" in spec:
            trial["repeat_id"] = spec["repeat_id"]

    fz_left_observed = [_round(x, 3) for x in observed_trace["fz_left_N"]]
    fz_right_observed = [_round(x, 3) for x in observed_trace["fz_right_N"]]
    fz_total_observed = [
        _round(left + right, 3)
        for left, right in zip(fz_left_observed, fz_right_observed)
    ]
    trial["observations"] = {
        "time_s": [_round(t, 2) for t in observed_trace["time_s"]],
        "fz_left_N": fz_left_observed,
        "fz_right_N": fz_right_observed,
        "fz_total_N": fz_total_observed,
        "bar_displacement_m": [_round(x, 6) for x in observed_trace["bar_displacement_m"]],
        "bar_velocity_m_s": [_round(x, 6) for x in observed_trace["bar_velocity_m_s"]],
    }
    trial["observed_events"] = {
        "weighing_start_time_s": 0.0,
        "weighing_end_time_s": 1.5,
        "movement_onset_time_s": _round(float(events["movement_onset_time_s"]), 3),
        "takeoff_time_s": _round(float(events["takeoff_time_s"]), 3),
    }
    trial["observed_summary"] = {
        key: _round(float(summary[key]), 6)
        for key in SUMMARY_KEYS
    }
    if float(spec.get("drive_asymmetry_alpha", 0.0)) != 0.0:
        trial["bilateral_measurements"] = _bilateral_measurements(observed_trace, events)
    _assert_trial_is_finite(trial)
    return trial


def _observed_resampled_trace(
    raw_trace: dict[str, Any],
    noise_seed: int,
    force_noise_sd_N: float,
    bar_displacement_noise_sd_m: float,
) -> dict[str, Any]:
    rng = SplitMix64(GENERATION_SEED ^ noise_seed)
    native_time = _float_list(raw_trace["time_s"])
    raw_left = _float_list(raw_trace["fz_left_N"])
    raw_right = _float_list(raw_trace["fz_right_N"])
    native_left: list[float] = []
    native_right: list[float] = []
    for left, right in zip(raw_left, raw_right):
        total = max(0.0, left + right + force_noise_sd_N * rng.normal())
        raw_total = left + right
        if raw_total > 0.0:
            scale = total / raw_total
            native_left_value = left * scale
            native_right_value = right * scale
        else:
            native_left_value = 0.0
            native_right_value = 0.0
        native_left.append(native_left_value)
        native_right.append(native_right_value)
    native_bar_disp = [
        float(x) + bar_displacement_noise_sd_m * rng.normal()
        for x in _float_list(raw_trace["bar_displacement_m"])
    ]

    fz_left = _resample_linear(native_time, native_left, COMMON_GRID_S)
    fz_right = _resample_linear(native_time, native_right, COMMON_GRID_S)
    fz_total = [left + right for left, right in zip(fz_left, fz_right)]
    bar_disp = _resample_linear(native_time, native_bar_disp, COMMON_GRID_S)
    bar_vel = _differentiate(COMMON_GRID_S, bar_disp)
    trace = {
        "time_s": list(COMMON_GRID_S),
        "fz_left_N": fz_left,
        "fz_right_N": fz_right,
        "fz_total_N": fz_total,
        "fnet_N": [0.0 for _ in COMMON_GRID_S],
        "root_z_m": _resample_linear(native_time, _float_list(raw_trace["root_z_m"]), COMMON_GRID_S),
        "root_x_m": _resample_linear(native_time, _float_list(raw_trace["root_x_m"]), COMMON_GRID_S),
        "root_pitch_rad": _resample_linear(native_time, _float_list(raw_trace["root_pitch_rad"]), COMMON_GRID_S),
        "bar_z_m": _resample_linear(native_time, _float_list(raw_trace["bar_z_m"]), COMMON_GRID_S),
        "bar_displacement_m": bar_disp,
        "bar_velocity_m_s": bar_vel,
        "lpt_tether_force_N": _resample_linear(native_time, _float_list(raw_trace["lpt_tether_force_N"]), COMMON_GRID_S),
        "left_foot_contact": _resample_nearest_bool(native_time, raw_trace["left_foot_contact"], COMMON_GRID_S),
        "right_foot_contact": _resample_nearest_bool(native_time, raw_trace["right_foot_contact"], COMMON_GRID_S),
    }
    return trace


def _integrate_segment(time_s: list[float], values: list[float], start: int, end: int) -> float:
    if end <= start:
        return 0.0
    return sum(
        0.5 * (float(values[i]) + float(values[i + 1]))
        * (float(time_s[i + 1]) - float(time_s[i]))
        for i in range(start, end)
    )


def _bilateral_measurements(
    trace: dict[str, Any],
    events: dict[str, Any],
) -> dict[str, Any]:
    """Return direct side-specific impulse diagnostics for the known excitation."""

    time_s = _float_list(trace["time_s"])
    root_z = _float_list(trace["root_z_m"])
    left = _float_list(trace["fz_left_N"])
    right = _float_list(trace["fz_right_N"])
    onset = int(events["movement_onset_index"])
    takeoff = int(events["takeoff_index"])
    propulsion_start = min(range(onset, takeoff + 1), key=lambda i: root_z[i])
    quiet = [i for i, t in enumerate(time_s) if 0.50 <= t <= 1.50]
    if not quiet:
        raise ValueError("no quiet samples for bilateral impulse diagnostics")
    left_quiet = sum(left[i] for i in quiet) / len(quiet)
    right_quiet = sum(right[i] for i in quiet) / len(quiet)
    left_net = [value - left_quiet for value in left]
    right_net = [value - right_quiet for value in right]
    return {
        "definition": "side force integrated after subtracting that side's quiet weighing mean",
        "units": "N*s",
        "braking_phase_start_time_s": _round(time_s[onset], 2),
        "braking_phase_end_time_s": _round(time_s[propulsion_start], 2),
        "propulsive_phase_start_time_s": _round(time_s[propulsion_start], 2),
        "propulsive_phase_end_time_s": _round(time_s[takeoff], 2),
        "left_braking_impulse_Ns": _round(_integrate_segment(time_s, left_net, onset, propulsion_start), 6),
        "right_braking_impulse_Ns": _round(_integrate_segment(time_s, right_net, onset, propulsion_start), 6),
        "left_propulsive_impulse_Ns": _round(_integrate_segment(time_s, left_net, propulsion_start, takeoff), 6),
        "right_propulsive_impulse_Ns": _round(_integrate_segment(time_s, right_net, propulsion_start, takeoff), 6),
    }


def _validation_descriptor_fields(spec: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "observation_noise_seed": int(spec["noise_seed"]),
        "force_noise_sd_N": _round(float(spec["force_noise_sd_N"]), 4),
        "bar_displacement_noise_sd_m": _round(float(spec["bar_displacement_noise_sd_m"]), 7),
        "family_focus": spec["family_focus"],
    }
    if "repeat_index" in spec:
        fields["repeat_index"] = int(spec["repeat_index"])
    return fields


def _validation_trial_specs() -> list[dict[str, Any]]:
    groups = [
        (
            "depth_timing_variants",
            "same-load countermovement-depth and braking/propulsion timing coverage",
            [
                (1.16, 0.94, 1.04, 71101, 1.3, 0.00013),
                (1.08, 1.06, 0.96, 71102, 1.4, 0.00013),
                (0.96, 0.94, 1.04, 71103, 1.4, 0.00014),
                (1.04, 1.06, 0.96, 71104, 1.5, 0.00014),
            ],
        ),
        (
            "depth_timing_repeatability",
            "same-load depth and phase-timing repeatability with independent measurement realizations",
            [
                (1.04, 1.06, 0.96, 71201, 1.4, 0.00014),
                (1.16, 0.94, 1.04, 71202, 1.4, 0.00014),
                (1.08, 1.06, 0.96, 71203, 1.5, 0.00015),
                (0.96, 0.94, 1.04, 71204, 1.5, 0.00015),
            ],
        ),
        (
            "force_platform_contact_variants",
            "same-load bilateral force-platform and contact coverage across phase variants",
            [
                (1.12, 0.96, 1.02, 71301, 1.5, 0.00015),
                (1.00, 1.04, 0.98, 71302, 1.5, 0.00015),
                (1.08, 0.96, 1.02, 71303, 1.6, 0.00016),
                (1.00, 1.04, 0.98, 71304, 1.6, 0.00016),
            ],
        ),
        (
            "bar_kinematics_variants",
            "same-load bar/LPT kinematics across phase variants",
            [
                (1.16, 1.04, 0.98, 71401, 1.5, 0.00015),
                (1.04, 0.96, 1.02, 71402, 1.5, 0.00015),
                (1.12, 1.04, 0.98, 71403, 1.6, 0.00016),
                (0.96, 0.96, 1.02, 71404, 1.6, 0.00016),
            ],
        ),
        (
            "force_platform_contact_repeatability",
            "same-load force-platform/contact repeatability with independent measurement realizations",
            [
                (1.04, 1.04, 0.98, 71501, 1.8, 0.00015),
                (1.12, 0.96, 1.02, 71502, 1.8, 0.00016),
                (1.00, 1.04, 0.98, 71503, 1.9, 0.00016),
                (1.08, 0.96, 1.02, 71504, 1.9, 0.00017),
            ],
        ),
        (
            "bar_kinematics_repeatability",
            "same-load bar/LPT kinematics repeatability with independent measurement realizations",
            [
                (1.08, 1.04, 0.98, 71601, 1.5, 0.00018),
                (0.96, 0.96, 1.02, 71602, 1.5, 0.00018),
                (1.16, 1.04, 0.98, 71603, 1.6, 0.00019),
                (1.00, 0.96, 1.02, 71604, 1.6, 0.00019),
            ],
        ),
        (
            "measurement_noise_repeats",
            "deterministic measurement-realization repeats across same-load mechanics conditions",
            [
                (1.12, 0.96, 1.02, 71701, 1.3, 0.00013),
                (1.00, 1.04, 0.98, 71702, 1.3, 0.00013),
                (1.08, 0.96, 1.02, 71703, 1.3, 0.00014),
                (1.04, 1.04, 0.98, 71704, 1.3, 0.00014),
            ],
        ),
        (
            "contact_timing_depth_variants",
            "same-load contact timing, depth, and phase-timing variation",
            [
                (1.16, 0.94, 1.04, 71801, 1.7, 0.00016),
                (1.08, 1.06, 0.96, 71802, 1.7, 0.00016),
                (0.96, 0.94, 1.04, 71803, 1.8, 0.00017),
                (1.04, 1.06, 0.96, 71804, 1.8, 0.00017),
            ],
        ),
    ]
    specs: list[dict[str, Any]] = []
    for group_name, family_focus, rows in groups:
        for row in rows:
            trial_index = len(specs) + 1
            spec = {
                "trial_id": f"validation_{trial_index:03d}",
                "trial_group": group_name,
                "family_focus": family_focus,
                "external_load_kg": PUBLIC_EXTERNAL_LOAD_KG,
                "depth_scale": row[0],
                "braking_duration_scale": row[1],
                "propulsion_duration_scale": row[2],
                "noise_seed": row[3],
                "force_noise_sd_N": row[4],
                "bar_displacement_noise_sd_m": row[5],
            }
            if group_name == "measurement_noise_repeats":
                spec["repeat_index"] = len([s for s in specs if s["trial_group"] == group_name]) + 1
            specs.append(spec)
    # Repeated active tuples are intentional noise repeats, never label-only
    # duplicates.  Give every member of a repeated tuple the same explicit ID;
    # the ID is excluded from active uniqueness by the audit code.
    by_active_tuple: dict[tuple[float, float, float, float], list[dict[str, Any]]] = {}
    for spec in specs:
        active = tuple(
            float(spec[key])
            for key in ("external_load_kg", "depth_scale", "braking_duration_scale", "propulsion_duration_scale")
        )
        by_active_tuple.setdefault(active, []).append(spec)
    repeat_number = 0
    for members in by_active_tuple.values():
        if len(members) > 1:
            repeat_number += 1
            repeat_id = f"mechanics_repeat_{repeat_number:02d}"
            for member in members:
                member["repeat_id"] = repeat_id
    if len(specs) != 32:
        raise AssertionError(f"expected 32 validation trials, got {len(specs)}")
    return specs


def _preprocessing_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "comparison_grid": {
            "sample_rate_hz": 100,
            "start_time_s": GRID_START_S,
            "end_time_s": GRID_END_S,
            "step_s": GRID_DT_S,
            "sample_count": GRID_SAMPLE_COUNT,
            "time_s": COMMON_GRID_S,
        },
        "resampling": {
            "convention": "Deterministic linear interpolation from plant.run_trial traces onto the common comparison grid.",
            "time_grid_rule": "Inclusive endpoints from 0.00 s to 3.60 s at 0.01 s spacing; times are rounded to two decimals.",
            "bar_velocity_m_s": "Computed by centered finite differences on the resampled bar displacement signal.",
        },
        "weighing_phase": {
            "duration_text": "1.50 s",
            "duration_s": 1.5,
            "stable_window_text": "Final stable 1.00 s of weighing.",
            "stable_window_s": 1.0,
            "Wsys": "Mean Fz_total over the final stable 1.00 s of the 1.50 s weighing phase.",
            "quiet_force_sd": "Population standard deviation of Fz_total over the same final stable 1.00 s weighing window.",
        },
        "force_definitions": {
            "Fz_left": "Source-authoritative left force-platform vertical force in N.",
            "Fz_right": "Source-authoritative right force-platform vertical force in N.",
            "Fz_total": "Bilateral force-platform vertical force, defined as Fz_left + Fz_right in N.",
            "Fnet": "Fz_total - Wsys.",
            "msys": "Wsys / 9.81.",
        },
        "event_rules": {
            "movement_onset": "First post-weighing sustained drop below Wsys minus max(5 x quiet_force_sd, 20 N), sustained for 0.050 s.",
            "takeoff": "First sustained no-foot-contact interval after the valid propulsive phase, sustained for 0.025 s.",
            "takeoff_rule_text": "sustained no-foot-contact",
        },
        "mechanics_validity_contract": {
            "primary_truth": "Validation derives validity from plant rollout force/contact/root/foot-clearance telemetry.",
            "phase_requirement": "All seven phase indices must be observed in the rollout.",
            "minimum_countermovement_depth_m": 0.08,
            "preferred_countermovement_depth_m": [0.10, 0.25],
            "minimum_no_contact_interval_s": 0.12,
            "maximum_impulse_flight_residual_s": 0.05,
            "minimum_flight_clearance_m": 0.001,
            "maximum_landing_rebound_m": 0.02,
            "maximum_posterior_drift_m": 0.08,
            "maximum_abs_torso_pitch_rad": 0.25,
            "maximum_abs_torso_pitch_rate_rad_s": 8.0,
            "lpt_rule": "Passive LPT/bar traces cannot override failed force/contact validity.",
        },
        "jump_height": {
            "hIM": "Impulse-momentum jump height computed as v_takeoff^2/(2g).",
            "g_m_s2": 9.81,
        },
        "lpt": {
            "interpretation": "LPT displacement and velocity are bar-only measurements and not COM measurements.",
            "bar_only_text": "bar-only, not COM",
        },
        "measurement_surface": {
            "force_units": "N",
            "bar_displacement_units": "m",
            "bar_velocity_units": "m/s",
            "force_aggregation": "fz_total_N equals fz_left_N + fz_right_N after deterministic resampling.",
            "force_realization": "The declared total-force noise realization is apportioned across the physical left/right force traces before resampling so aggregation remains exact.",
        },
    }


def _dataset_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generation_seed": GENERATION_SEED,
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
        },
        "public_trial_count": 6,
        "validation_trial_count": 32,
        "public_external_load_kg": PUBLIC_EXTERNAL_LOAD_KG,
        "identification_trial_ids": [spec["trial_id"] for spec in PUBLIC_TRIAL_SPECS],
        "qualified_known_excitations": {
            "synthetic_interlimb_drive_asymmetry_alpha": QUALIFIED_ASYMMETRY_ALPHA,
        },
        "signal_keys": {
            "observations": OBSERVATION_KEYS,
            "observed_events": EVENT_KEYS,
            "observed_summary": SUMMARY_KEYS,
        },
        "optional_trial_sections": {
            "bilateral_measurements": "Direct left/right braking and propulsive impulse diagnostics for the qualified known excitation.",
        },
        "generated_file_sha256": {},
    }


def _resample_linear(xs: list[float], ys: list[float], grid: list[float]) -> list[float]:
    if len(xs) != len(ys) or not xs:
        raise ValueError("resampling inputs must be equal-length nonempty lists")
    out: list[float] = []
    j = 0
    last = len(xs) - 1
    for target in grid:
        while j < last - 1 and xs[j + 1] < target:
            j += 1
        if target <= xs[0]:
            out.append(float(ys[0]))
        elif target >= xs[last]:
            out.append(float(ys[last]))
        else:
            x0 = xs[j]
            x1 = xs[j + 1]
            y0 = ys[j]
            y1 = ys[j + 1]
            if x1 <= x0:
                out.append(float(y0))
            else:
                alpha = (target - x0) / (x1 - x0)
                out.append(float(y0 + alpha * (y1 - y0)))
    return out


def _resample_nearest_bool(xs: list[float], values: Any, grid: list[float]) -> list[bool]:
    vals = [bool(x) for x in values]
    out: list[bool] = []
    j = 0
    last = len(xs) - 1
    for target in grid:
        while j < last - 1 and xs[j + 1] < target:
            j += 1
        if j >= last:
            out.append(vals[last])
            continue
        before = abs(target - xs[j])
        after = abs(xs[j + 1] - target)
        out.append(vals[j] if before <= after else vals[j + 1])
    return out


def _differentiate(time_s: list[float], values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    out = [0.0 for _ in values]
    for i in range(1, len(values) - 1):
        dt = time_s[i + 1] - time_s[i - 1]
        out[i] = (values[i + 1] - values[i - 1]) / dt if dt > 0.0 else 0.0
    first_dt = time_s[1] - time_s[0]
    last_dt = time_s[-1] - time_s[-2]
    out[0] = (values[1] - values[0]) / first_dt if first_dt > 0.0 else 0.0
    out[-1] = (values[-1] - values[-2]) / last_dt if last_dt > 0.0 else 0.0
    return out


def _float_list(values: Any) -> list[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(x) for x in values]


def _round(value: float, digits: int) -> float:
    rounded = round(float(value), digits)
    if rounded == 0 or abs(rounded) < 0.5 * (10 ** -digits):
        return 0.0
    return rounded


def _assert_trial_is_finite(trial: dict[str, Any]) -> None:
    observations = trial["observations"]
    lengths = {len(observations[key]) for key in OBSERVATION_KEYS}
    if len(lengths) != 1 or next(iter(lengths)) != GRID_SAMPLE_COUNT:
        raise ValueError(f"{trial['trial_id']} has inconsistent observation lengths")
    for key in OBSERVATION_KEYS:
        for value in observations[key]:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{trial['trial_id']} nonfinite observation {key}")
    for section_name in ("observed_events", "observed_summary"):
        for key, value in trial[section_name].items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{trial['trial_id']} nonfinite {section_name} {key}")
    for key, value in trial.get("bilateral_measurements", {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
            raise ValueError(f"{trial['trial_id']} nonfinite bilateral measurement {key}")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
