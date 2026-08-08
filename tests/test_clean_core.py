from __future__ import annotations

import ast
from dataclasses import fields
import importlib.util
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest


TASK_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = TASK_ROOT / "src" / "loaded_cmj"
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
TELEMETRY_FIELDS = (
    "time_s",
    "qpos",
    "qvel",
    "qacc",
    "ctrl",
    "actuator_force",
    "root_position_m",
    "root_velocity_m_s",
    "bar_position_m",
    "bar_velocity_m_s",
    "contact_count",
    "contacts",
    "total_relevant_vertical_force_N",
    "total_mass_kg",
    "com_position_m",
    "com_velocity_m_s",
    "linear_momentum_kg_m_s",
    "kinetic_energy_J",
    "potential_energy_J",
    "actuator_power_W",
    "finite_state",
)


def _load_plant() -> Any:
    path = PACKAGE_ROOT / "plant.py"
    spec = importlib.util.spec_from_file_location("cca01_test_plant", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def plant() -> Any:
    return _load_plant()


@pytest.fixture(scope="module")
def model(plant: Any) -> mujoco.MjModel:
    return plant.build_model(plant.default_params())


def _state(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate(
        (
            np.array([data.time]),
            data.qpos.copy(),
            data.qvel.copy(),
            data.qacc.copy(),
            data.ctrl.copy(),
            data.actuator_force.copy(),
        )
    )


def _rollout(plant: Any, model: mujoco.MjModel, controls: list[np.ndarray], observe: bool) -> np.ndarray:
    data = plant.reset_data(model)
    trajectory = []
    for control in controls:
        plant._clean_core.apply_control(model, data, control)
        plant._clean_core.step(model, data)
        if observe:
            telemetry = plant._clean_core.collect_telemetry(model, data)
            assert telemetry.finite_state
        trajectory.append(_state(data))
    return np.asarray(trajectory)


def test_p0_model_compilation_reset_and_algebraic_invariants(plant: Any, model: mujoco.MjModel) -> None:
    assert {key: int(getattr(model, key)) for key in EXPECTED_COUNTS} == EXPECTED_COUNTS
    data = plant.reset_data(model)
    assert data.time == 0.0
    assert np.isfinite(data.qpos).all()
    assert np.array_equal(data.qvel, np.zeros(model.nv))
    assert np.all(model.body_mass[1:] > 0.0)
    assert np.all(model.body_inertia[1:] > 0.0)
    limited = np.asarray(model.jnt_limited, dtype=bool)
    assert np.all(np.isfinite(model.jnt_range[limited]))
    assert np.all(model.jnt_range[limited, 0] < model.jnt_range[limited, 1])
    hinge_or_slide = np.isin(
        model.jnt_type,
        [int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)],
    )
    assert np.allclose(np.linalg.norm(model.jnt_axis[hinge_or_slide], axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_p1_neutral_rollout_has_finite_mechanical_diagnostics(plant: Any, model: mujoco.MjModel) -> None:
    data = plant.reset_data(model)
    zero = np.zeros(model.nu)
    for _ in range(100):
        plant._clean_core.apply_control(model, data, zero)
        plant._clean_core.step(model, data)
    telemetry = plant._clean_core.collect_telemetry(model, data)
    assert telemetry.finite_state
    assert telemetry.kinetic_energy_J >= 0.0
    assert telemetry.total_mass_kg > 0.0
    assert np.isfinite(telemetry.potential_energy_J)
    assert np.array_equal(data.qfrc_applied, np.zeros(model.nv))
    assert np.array_equal(data.xfrc_applied, np.zeros_like(data.xfrc_applied))


def test_p2_deterministic_replay_is_exact_on_captured_platform(plant: Any, model: mujoco.MjModel) -> None:
    controls = [np.zeros(model.nu) for _ in range(100)]
    trajectories = [_rollout(plant, model, controls, observe=False) for _ in range(3)]
    assert np.array_equal(trajectories[0], trajectories[1], equal_nan=True)
    assert np.array_equal(trajectories[0], trajectories[2], equal_nan=True)


def test_p3_bound_actuator_perturbation_changes_state_only_after_step(plant: Any, model: mujoco.MjModel) -> None:
    data = plant.reset_data(model)
    initial_qpos = data.qpos.copy()
    control = np.zeros(model.nu)
    control[0] = 50.0
    plant._clean_core.apply_control(model, data, control)
    assert np.array_equal(data.qpos, initial_qpos)
    for _ in range(30):
        plant._clean_core.step(model, data)
        plant._clean_core.apply_control(model, data, control)
    baseline = _rollout(plant, model, [np.zeros(model.nu) for _ in range(30)], observe=False)
    assert np.any(np.abs(data.qpos - baseline[-1, 1 : 1 + model.nq]) > 0.0)
    assert np.any(np.abs(data.actuator_force) > 0.0)


def test_p4_contact_telemetry_uses_actual_eligible_contacts_and_world_frame(plant: Any, model: mujoco.MjModel) -> None:
    data = plant.reset_data(model)
    telemetry = plant._clean_core.collect_telemetry(model, data)
    assert telemetry.contact_count == data.ncon > 0
    relevant = [contact for contact in telemetry.contacts if contact.relevant_foot_geom_id is not None]
    assert relevant
    independent_total = 0.0
    for item in relevant:
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, item.index, wrench)
        frame = np.asarray(data.contact[item.index].frame).reshape(3, 3)
        world = frame.T @ wrench[:3]
        assert item.penetration_m == max(0.0, -item.distance_m)
        assert np.allclose(item.world_force_N, world, atol=1e-12, rtol=0.0)
        independent_total += abs(float(world[2]))
    assert telemetry.total_relevant_vertical_force_N == pytest.approx(independent_total, abs=1e-12, rel=0.0)


def test_p5_bar_telemetry_is_live_mujoco_state_with_mechanical_coupling(plant: Any, model: mujoco.MjModel) -> None:
    data = plant.reset_data(model)
    telemetry = plant._clean_core.collect_telemetry(model, data)
    bar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bar")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bar_lpt_site")
    assert bar_id >= 0 and site_id >= 0
    assert model.body_mass[bar_id] > 0.0
    assert np.all(model.body_inertia[bar_id] > 0.0)
    assert np.array_equal(telemetry.bar_position_m, data.site_xpos[site_id])
    for tendon in ("left_hand_grip", "right_hand_grip"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon) >= 0
    for joint in ("bar_rack_x", "bar_rack_z", "bar_rack_pitch"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) >= 0


def test_p6_telemetry_is_observationally_pure(plant: Any, model: mujoco.MjModel) -> None:
    controls = [np.zeros(model.nu) for _ in range(100)]
    unobserved = _rollout(plant, model, controls, observe=False)
    observed = _rollout(plant, model, controls, observe=True)
    assert np.array_equal(unobserved, observed, equal_nan=True)


def test_p7_only_reset_writes_state_and_observation_does_not_mutate_it(plant: Any, model: mujoco.MjModel) -> None:
    assignments: list[tuple[str, int, str]] = []
    for path in (PACKAGE_ROOT / "clean_core.py", PACKAGE_ROOT / "plant.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else []
            for target in targets:
                rendered = ast.unparse(target)
                if ".qpos" in rendered or ".qvel" in rendered:
                    assignments.append((path.name, node.lineno, rendered))
    ordered_assignments = sorted(assignments, key=lambda item: (item[0], item[1]))
    assert [(path, target) for path, _, target in ordered_assignments] == [
        ("clean_core.py", "data.qpos[int(model.jnt_qposadr[joint_id])]") ,
        ("clean_core.py", "data.qvel[:]") ,
    ]
    data = plant.reset_data(model)
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    plant._clean_core.collect_telemetry(model, data)
    assert np.array_equal(data.qpos, qpos)
    assert np.array_equal(data.qvel, qvel)
    assert np.array_equal(data.ctrl, ctrl)


def test_telemetry_schema_and_legacy_adapter_compatibility(plant: Any, model: mujoco.MjModel) -> None:
    assert tuple(field.name for field in fields(plant._clean_core.PlantTelemetry)) == TELEMETRY_FIELDS
    assert not any("imu" in field.lower() or "fusion" in field.lower() for field in TELEMETRY_FIELDS)
    data = plant.reset_data(model)
    telemetry = plant._clean_core.collect_telemetry(model, data)
    assert telemetry.qpos.shape == (model.nq,)
    assert telemetry.qvel.shape == telemetry.qacc.shape == (model.nv,)
    assert telemetry.ctrl.shape == telemetry.actuator_force.shape == (model.nu,)
    assert telemetry.root_position_m.shape == telemetry.root_velocity_m_s.shape == (3,)
    assert telemetry.bar_position_m.shape == telemetry.bar_velocity_m_s.shape == (3,)
    assert plant.build_model(plant.default_params()).nq == model.nq
