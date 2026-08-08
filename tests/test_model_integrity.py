from __future__ import annotations

import hashlib

import mujoco
import numpy as np

from loaded_cmj.model import build_model, compiled_model_signature
from loaded_cmj.parameters import default_parameters, flatten_parameters


EXPECTED_COUNTS = {
    "nq": 21,
    "nv": 21,
    "nu": 6,
    "nbody": 19,
    "njnt": 21,
    "ngeom": 28,
    "ntendon": 4,
    "nsensor": 10,
    "npair": 6,
}


def test_compiled_model_contract() -> None:
    model = build_model(default_parameters())
    signature = compiled_model_signature(model)
    assert {key: signature[key] for key in EXPECTED_COUNTS} == EXPECTED_COUNTS
    assert signature["timestep_s"] == 0.002
    assert signature["integrator"] == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    assert signature["iterations"] == 100
    assert signature["tolerance"] == 1e-10
    assert signature["bodies"][1] == "pelvis"
    assert signature["joints"][:3] == ["root_x", "root_z", "root_pitch"]
    assert signature["actuators"] == [
        "left_hip_torque", "left_knee_torque", "left_ankle_torque",
        "right_hip_torque", "right_knee_torque", "right_ankle_torque",
    ]


def test_parameter_authority_is_full_source_vector() -> None:
    names, values, lower, upper = flatten_parameters(default_parameters())
    assert len(names) == len(values) == len(lower) == len(upper) == 27
    assert names[:4] == [
        "body_mass_kg",
        "joint_stiffness_Nm_rad[0]",
        "joint_stiffness_Nm_rad[1]",
        "joint_stiffness_Nm_rad[2]",
    ]
    assert all(lo <= value <= hi for value, lo, hi in zip(values, lower, upper))


def test_mjcf_is_the_directly_ported_asset() -> None:
    expected_sha256 = "54e092b2aefd664af64d1848a7cca0e82d1205156b0c79c00f9dbc70600f87e5"
    path = __import__("pathlib").Path(__file__).resolve().parents[1] / "assets" / "loaded_cmj_model.xml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
