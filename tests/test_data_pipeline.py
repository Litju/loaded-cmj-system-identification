from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from loaded_cmj.dataset import generate_dataset, load_dataset, load_trial, load_trials
from loaded_cmj.identification import DEFAULT_FIT_COORDINATES, identify_parameters
from loaded_cmj.parameters import default_parameters, load_named_parameters, parameter_schema
from loaded_cmj.preprocessing import comparison_grid, preprocess_observations
from loaded_cmj.validation import _physical_metrics, validate_trial
from loaded_cmj import plant


PUBLIC_TRIAL_IDS = [
    "20kg_nominal_a",
    "20kg_nominal_b",
    "20kg_depth",
    "20kg_timing",
    "20kg_depth_timing",
    "20kg_bilateral_asymmetry",
]


def _sha256_float64(values: object) -> str:
    array = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest()


def _sha256_bool(values: object) -> str:
    array = np.asarray(values, dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest()


QPOS_KEYS = {
    "root_x": "root_x_m", "root_z": "root_z_m", "root_pitch": "root_pitch_rad",
    "lumbar_pitch": "lumbar_pitch_rad", "bar_rack_x": "bar_rack_x_m",
    "bar_rack_z": "bar_rack_z_m", "bar_rack_pitch": "bar_rack_pitch_rad",
    "left_shoulder": "left_shoulder_rad", "left_elbow": "left_elbow_rad",
    "right_shoulder": "right_shoulder_rad", "right_elbow": "right_elbow_rad",
    "left_hip": "left_hip_rad", "left_knee": "left_knee_rad", "left_ankle": "left_ankle_rad",
    "left_forefoot_rocker": "left_forefoot_rocker_rad", "left_mtp": "left_mtp_rad",
    "right_hip": "right_hip_rad", "right_knee": "right_knee_rad", "right_ankle": "right_ankle_rad",
    "right_forefoot_rocker": "right_forefoot_rocker_rad", "right_mtp": "right_mtp_rad",
}
QVEL_KEYS = {
    "root_x": "root_x_velocity_m_s", "root_z": "root_z_velocity_m_s",
    "root_pitch": "root_pitch_rate_rad_s", "lumbar_pitch": "lumbar_pitch_rate_rad_s",
    "bar_rack_x": "bar_rack_x_velocity_m_s", "bar_rack_z": "bar_rack_z_velocity_m_s",
    "bar_rack_pitch": "bar_rack_pitch_velocity_rad_s",
    "left_shoulder": "left_shoulder_velocity_rad_s", "left_elbow": "left_elbow_velocity_rad_s",
    "right_shoulder": "right_shoulder_velocity_rad_s", "right_elbow": "right_elbow_velocity_rad_s",
    "left_hip": "left_hip_velocity_rad_s", "left_knee": "left_knee_velocity_rad_s",
    "left_ankle": "left_ankle_velocity_rad_s", "left_forefoot_rocker": "left_forefoot_rocker_velocity_rad_s",
    "left_mtp": "left_mtp_velocity_rad_s", "right_hip": "right_hip_velocity_rad_s",
    "right_knee": "right_knee_velocity_rad_s", "right_ankle": "right_ankle_velocity_rad_s",
    "right_forefoot_rocker": "right_forefoot_rocker_velocity_rad_s", "right_mtp": "right_mtp_velocity_rad_s",
}


def test_public_and_validation_fixtures_are_explicit() -> None:
    identification = load_dataset("identification")
    validation = load_dataset("validation")
    assert identification["split"] == "public"
    assert validation["split"] == "validation"
    assert len(identification["trials"]) == 6
    assert len(validation["trials"]) == 32
    assert [trial["trial_id"] for trial in identification["trials"]] == PUBLIC_TRIAL_IDS
    assert all(trial["external_load_kg"] == 20.0 for trial in identification["trials"])
    assert all(trial["external_load_kg"] == 20.0 for trial in validation["trials"])
    assert all(trial["bar_load_kg"] == 20.0 for trial in identification["trials"])
    assert all(trial["bar_load_kg"] == 20.0 for trial in validation["trials"])
    assert all(trial["trial_id"].startswith("validation_") for trial in validation["trials"])


def test_preprocessing_preserves_units_and_grid() -> None:
    trial = load_trial("20kg_nominal_a", split="identification")
    processed = preprocess_observations(trial["observations"])
    assert len(comparison_grid()) == 361
    assert np.isclose(processed["msys_kg"], processed["Wsys_N"] / 9.81)
    assert len(processed["fz_total_N"]) == len(trial["observations"]["time_s"])
    assert np.allclose(
        np.asarray(processed["fnet_N"]),
        np.asarray(processed["fz_total_N"]) - processed["Wsys_N"],
    )


def test_bilateral_force_surface_aggregates_exactly_and_lpt_is_bar_only() -> None:
    preprocessing = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "preprocessing.json").read_text()
    )
    assert "fz_left_N" in preprocessing["measurement_surface"]["force_aggregation"]
    assert preprocessing["lpt"]["interpretation"] == (
        "LPT displacement and velocity are bar-only measurements and not COM measurements."
    )
    assert preprocessing["jump_height"]["hIM"] == (
        "Impulse-momentum jump height computed as v_takeoff^2/(2g)."
    )
    for split in ("identification", "validation"):
        for trial in load_trials(split):
            observations = trial["observations"]
            assert np.allclose(
                np.asarray(observations["fz_total_N"]),
                np.asarray(observations["fz_left_N"]) + np.asarray(observations["fz_right_N"]),
                atol=1e-12,
                rtol=0.0,
            )


def test_nominal_20kg_regression_matches_gate0_fingerprints() -> None:
    result = plant.run_trial(load_named_parameters("synthetic_reference"), {"external_load_kg": 20.0})
    traces = result["traces"]
    assert result["valid"] is True
    assert result["events"]["movement_onset_time_s"] == 1.568
    assert result["events"]["takeoff_time_s"] == 2.17
    assert result["events"]["landing_time_s"] == 2.578
    assert _sha256_float64(traces["fz_left_N"]) == "b6e062adb2448ad5a86bcb3e2103efab1ea0885546e980e39b815cf7a0476d2e"
    assert _sha256_float64(traces["fz_right_N"]) == "46f7ea5bb2c00872ab2f205ac177e84fdac916bb001aeb0f34ce71cccb7d166e"
    assert _sha256_float64(traces["fz_total_N"]) == "2d406ec29eb8e27c95be2a2f6aa4b1e8ff29e6a34222b07f353e3c20bbcefc12"
    assert _sha256_float64(traces["bar_displacement_m"]) == "03bf2fdfa385a7fb55e402712250ef95643782f8435864fc5bdbde2157e7e218"
    assert _sha256_float64(traces["bar_velocity_m_s"]) == "c785de4142bd4eb49fde005d684335349240045ce5e3b34cd47eb94f0cdb732e"
    assert _sha256_bool(traces["left_foot_contact"]) == "ff9170c5ee91272a021a873232f4d16c40401a51b6a7ac03658f104a385593fe"
    assert _sha256_bool(traces["right_foot_contact"]) == "ff9170c5ee91272a021a873232f4d16c40401a51b6a7ac03658f104a385593fe"
    model = plant.build_model(load_named_parameters("synthetic_reference"), {"external_load_kg": 20.0})
    joint_names = [plant.mujoco.mj_id2name(model, plant.mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    qpos = np.column_stack([traces[QPOS_KEYS[name]] for name in joint_names])
    qvel = np.column_stack([traces[QVEL_KEYS[name]] for name in joint_names])
    assert _sha256_float64(qpos) == "84ffc29e2b9d5c99ec2e6224cc2107751455500c215489359e916585ab3db26c"
    assert _sha256_float64(qvel) == "453e6309f8088f10c34c55ac6ef31d14b430bed3f9c6b7f874067a6e651bf920"


def test_qualified_asymmetry_is_known_and_mechanically_valid() -> None:
    trial = load_trial("20kg_bilateral_asymmetry", split="identification")
    bilateral = trial["bilateral_measurements"]
    assert bilateral["units"] == "N*s"
    assert bilateral["left_propulsive_impulse_Ns"] > bilateral["right_propulsive_impulse_Ns"]
    result = plant.run_trial(load_named_parameters("synthetic_reference"), trial)
    assert result["valid"] is True
    assert result["diagnostics"]["drive_asymmetry_alpha"] == 0.02
    assert result["diagnostics"]["drive_asymmetry_clip_count"] == 0
    physical = _physical_metrics(result)
    assert physical["mechanics_valid"] is True
    assert all("asymmetry" not in name.lower() for name in DEFAULT_FIT_COORDINATES)
    propulsive_start = min(
        range(result["events"]["movement_onset_index"], result["events"]["takeoff_index"] + 1),
        key=lambda index: result["traces"]["root_z_m"][index],
    )
    end = result["events"]["takeoff_index"]
    time = np.asarray(result["traces"]["time_s"])[propulsive_start : end + 1]
    left = np.asarray(result["traces"]["fz_left_N"])[propulsive_start : end + 1]
    right = np.asarray(result["traces"]["fz_right_N"])[propulsive_start : end + 1]
    assert abs(float(np.trapezoid(left - right, time))) > 5.0


def test_dataset_regeneration_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(output_dir=first)
    generate_dataset(output_dir=second)
    relative_files = (
        "data/identification_trials.json",
        "data/validation_trials.json",
        "data/dataset_manifest.json",
        "configs/preprocessing.json",
        "configs/synthetic_reference.json",
    )
    for relative in relative_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_manifest_hashes_reproduce() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data" / "dataset_manifest.json").read_text())
    for relative, expected in manifest["generated_file_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected


def test_identification_pipeline_executes_without_rollout_failures() -> None:
    result = identify_parameters(
        load_trials("identification")[:1],
        parameter_names=("encoder_offset_m",),
        max_nfev=2,
        residual_sample_count=8,
    )
    assert result["rollout_failures"] == 0
    assert result["parameter_names"] == ["encoder_offset_m"]
    assert np.isfinite(result["cost"])


def test_validation_pipeline_preserves_phase_and_recovery_contract() -> None:
    result = validate_trial(
        load_named_parameters("synthetic_reference"),
        load_trial("20kg_nominal_a", split="identification"),
    )
    assert result["valid"] is True
    assert result["physical"]["mechanics_valid"] is True
    assert result["predicted_events"]["phase_order_valid"] is True
    assert result["predicted_events"]["sustained_no_foot_contact"] is True
    assert result["physical"]["mechanics_gates"]["touchdown_triggered_landing"] is True


def test_schema_has_units_and_exact_nominal_keys() -> None:
    schema = parameter_schema()
    nominal = default_parameters()
    assert set(schema["required"]) == set(nominal)
    assert schema["properties"]["body_mass_kg"]["units"] == "kg"
    assert schema["properties"]["bar_attachment_offset_m"]["units"] == "m"
    assert all("units" in definition for definition in schema["properties"].values())
    assert not any("asymmetry" in name.lower() for name in schema["required"])


def test_undefined_cop_samples_are_not_rollout_failures() -> None:
    result = {
        "traces": {
            "time_s": [0.0],
            "fz_total_N": [1.0],
            "root_z_m": [1.0],
            "root_x_m": [0.0],
            "foot_clearance_m": [0.0],
            "root_pitch_rad": [0.0],
            "root_pitch_rate_rad_s": [0.0],
            "lpt_tether_force_N": [0.0],
            "phase_index": [0],
            "left_foot_contact": [False],
            "right_foot_contact": [False],
            "cop_x_m": [float("nan")],
        },
        "events": {
            "movement_onset_index": 0,
            "takeoff_index": None,
            "landing_index": None,
        },
        "summary": {},
    }
    assert _physical_metrics(result)["finite"] is True
