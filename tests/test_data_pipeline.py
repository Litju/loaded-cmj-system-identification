from __future__ import annotations

import numpy as np

from loaded_cmj.dataset import load_dataset, load_trial, load_trials
from loaded_cmj.parameters import default_parameters, parameter_schema
from loaded_cmj.preprocessing import comparison_grid, preprocess_observations
from loaded_cmj.validation import _physical_metrics


def test_public_and_validation_fixtures_are_explicit() -> None:
    identification = load_dataset("identification")
    validation = load_dataset("validation")
    assert identification["split"] == "public"
    assert validation["split"] == "validation"
    assert len(identification["trials"]) == 6
    assert len(validation["trials"]) == 32
    assert all(trial["trial_id"].startswith("public_") for trial in identification["trials"])
    assert all(trial["trial_id"].startswith("validation_") for trial in validation["trials"])


def test_preprocessing_preserves_units_and_grid() -> None:
    trial = load_trial("public_001", split="identification")
    processed = preprocess_observations(trial["observations"])
    assert len(comparison_grid()) == 361
    assert np.isclose(processed["msys_kg"], processed["Wsys_N"] / 9.81)
    assert len(processed["fz_total_N"]) == len(trial["observations"]["time_s"])
    assert np.allclose(
        np.asarray(processed["fnet_N"]),
        np.asarray(processed["fz_total_N"]) - processed["Wsys_N"],
    )


def test_schema_has_units_and_exact_nominal_keys() -> None:
    schema = parameter_schema()
    nominal = default_parameters()
    assert set(schema["required"]) == set(nominal)
    assert schema["properties"]["body_mass_kg"]["units"] == "kg"
    assert schema["properties"]["bar_attachment_offset_m"]["units"] == "m"


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
