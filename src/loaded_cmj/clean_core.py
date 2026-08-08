"""Deterministic MuJoCo plant primitives for the loaded-CMJ study.

This module owns model compilation, data creation, reset, control writes,
stepping, and observation-only telemetry.  It does not own the CMJ controller,
phase logic, datasets, or sensor models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


FOOT_CONTACT_GEOMS = (
    "left_heel",
    "left_forefoot",
    "left_toe",
    "right_heel",
    "right_forefoot",
    "right_toe",
)


@dataclass(frozen=True)
class ModelConfiguration:
    xml_path: Path
    post_compile: Callable[[mujoco.MjModel], None] | None = None


@dataclass(frozen=True)
class TelemetryContext:
    root_body: str = "pelvis"
    bar_body: str = "bar"
    bar_site: str = "bar_lpt_site"
    force_plate_geom: str = "force_plate"
    foot_contact_geoms: tuple[str, ...] = FOOT_CONTACT_GEOMS


@dataclass(frozen=True)
class ContactTelemetry:
    index: int
    geom1_id: int
    geom2_id: int
    relevant_foot_geom_id: int | None
    distance_m: float
    penetration_m: float
    frame: tuple[float, ...]
    contact_force: tuple[float, ...]
    world_force_N: tuple[float, float, float]


@dataclass(frozen=True)
class PlantTelemetry:
    time_s: float
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    ctrl: np.ndarray
    actuator_force: np.ndarray
    root_position_m: np.ndarray
    root_velocity_m_s: np.ndarray
    bar_position_m: np.ndarray
    bar_velocity_m_s: np.ndarray
    contact_count: int
    contacts: tuple[ContactTelemetry, ...]
    total_relevant_vertical_force_N: float
    total_mass_kg: float
    com_position_m: np.ndarray
    com_velocity_m_s: np.ndarray
    linear_momentum_kg_m_s: np.ndarray
    kinetic_energy_J: float
    potential_energy_J: float
    actuator_power_W: float
    finite_state: bool


def build_model(configuration: ModelConfiguration) -> mujoco.MjModel:
    """Compile one committed MJCF model and apply bounded model parameters."""

    model = mujoco.MjModel.from_xml_path(str(configuration.xml_path))
    if configuration.post_compile is not None:
        configuration.post_compile(model)
    return model


def create_data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_condition: Mapping[str, float],
) -> None:
    """Reset and write the only legal production qpos/qvel initial state."""

    mujoco.mj_resetData(model, data)
    for joint_name, value in initial_condition.items():
        joint_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if int(model.jnt_type[joint_id]) not in (
            int(mujoco.mjtJoint.mjJNT_SLIDE),
            int(mujoco.mjtJoint.mjJNT_HINGE),
        ):
            raise ValueError(f"initial condition requires scalar joint: {joint_name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = _finite_scalar(value, joint_name)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    _require_finite_state(data)


def apply_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control: Sequence[float] | np.ndarray,
) -> None:
    values = np.asarray(control, dtype=np.float64)
    if values.shape != (model.nu,):
        raise ValueError(f"control shape must be ({model.nu},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("control must contain only finite values")
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    if np.any(values[limited] < ranges[limited, 0]) or np.any(values[limited] > ranges[limited, 1]):
        raise ValueError("control exceeds a compiled actuator ctrlrange")
    data.ctrl[:] = values


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_step(model, data)
    _require_finite_state(data)


def collect_telemetry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    context: TelemetryContext = TelemetryContext(),
) -> PlantTelemetry:
    """Read a causal post-step snapshot without changing plant state or control."""

    root_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, context.root_body)
    bar_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, context.bar_body)
    bar_site_id = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, context.bar_site)
    plate_id = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, context.force_plate_geom)
    foot_ids = {
        _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for geom_name in context.foot_contact_geoms
    }

    contacts, total_fz = _contacts(model, data, plate_id, foot_ids)
    root_velocity = _body_linear_velocity(model, data, root_id)
    bar_velocity = _body_linear_velocity(model, data, bar_id)
    masses = np.asarray(model.body_mass, dtype=np.float64)
    total_mass = float(np.sum(masses))
    body_positions = np.asarray(data.xipos, dtype=np.float64)
    body_velocities = np.vstack(
        [_body_linear_velocity(model, data, body_id) for body_id in range(model.nbody)]
    )
    momentum = np.sum(masses[:, None] * body_velocities, axis=0)
    com_position = np.sum(masses[:, None] * body_positions, axis=0) / total_mass
    com_velocity = momentum / total_mass
    full_mass = np.empty((model.nv, model.nv), dtype=np.float64)
    # MuJoCo 3.x Python bindings populate the full matrix from MjData directly;
    # older bindings accepted the sparse qM/M buffer as the third argument.
    try:
        mujoco.mj_fullM(model, data, full_mass)
    except TypeError:
        mass_buffer = getattr(data, "qM", getattr(data, "M", None))
        if mass_buffer is None:
            raise
        mujoco.mj_fullM(model, full_mass, mass_buffer)
    kinetic = 0.5 * float(data.qvel @ full_mass @ data.qvel)
    potential = float(np.sum(masses * (body_positions @ -np.asarray(model.opt.gravity))))
    actuator_power = float(np.dot(data.actuator_force, data.actuator_velocity))

    arrays = {
        "qpos": _snapshot(data.qpos),
        "qvel": _snapshot(data.qvel),
        "qacc": _snapshot(data.qacc),
        "ctrl": _snapshot(data.ctrl),
        "actuator_force": _snapshot(data.actuator_force),
        "root_position_m": _snapshot(data.xpos[root_id]),
        "root_velocity_m_s": _snapshot(root_velocity),
        "bar_position_m": _snapshot(data.site_xpos[bar_site_id]),
        "bar_velocity_m_s": _snapshot(bar_velocity),
        "com_position_m": _snapshot(com_position),
        "com_velocity_m_s": _snapshot(com_velocity),
        "linear_momentum_kg_m_s": _snapshot(momentum),
    }
    scalars = (
        float(data.time), total_fz, total_mass, kinetic, potential, actuator_power
    )
    finite = all(np.isfinite(value).all() for value in arrays.values()) and np.isfinite(scalars).all()
    return PlantTelemetry(
        time_s=float(data.time),
        qpos=arrays["qpos"],
        qvel=arrays["qvel"],
        qacc=arrays["qacc"],
        ctrl=arrays["ctrl"],
        actuator_force=arrays["actuator_force"],
        root_position_m=arrays["root_position_m"],
        root_velocity_m_s=arrays["root_velocity_m_s"],
        bar_position_m=arrays["bar_position_m"],
        bar_velocity_m_s=arrays["bar_velocity_m_s"],
        contact_count=int(data.ncon),
        contacts=contacts,
        total_relevant_vertical_force_N=total_fz,
        total_mass_kg=total_mass,
        com_position_m=arrays["com_position_m"],
        com_velocity_m_s=arrays["com_velocity_m_s"],
        linear_momentum_kg_m_s=arrays["linear_momentum_kg_m_s"],
        kinetic_energy_J=kinetic,
        potential_energy_J=potential,
        actuator_power_W=actuator_power,
        finite_state=bool(finite),
    )


def _contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plate_id: int,
    foot_ids: set[int],
) -> tuple[tuple[ContactTelemetry, ...], float]:
    result = []
    total_vertical = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, wrench)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        world_force = frame.T @ wrench[:3]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        relevant = geom2 if geom1 == plate_id and geom2 in foot_ids else None
        if geom2 == plate_id and geom1 in foot_ids:
            relevant = geom1
        if relevant is not None:
            total_vertical += abs(float(world_force[2]))
        result.append(ContactTelemetry(
            index=index,
            geom1_id=geom1,
            geom2_id=geom2,
            relevant_foot_geom_id=relevant,
            distance_m=float(contact.dist),
            penetration_m=max(0.0, -float(contact.dist)),
            frame=tuple(float(value) for value in frame.ravel()),
            contact_force=tuple(float(value) for value in wrench),
            world_force_N=tuple(float(value) for value in world_force),
        ))
    return tuple(result), total_vertical


def _body_linear_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> np.ndarray:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
    )
    return velocity[3:].copy()


def _required_id(model: mujoco.MjModel, object_type: Any, object_name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, object_name))
    if object_id < 0:
        raise ValueError(f"required MuJoCo object not found: {object_name}")
    return object_id


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"initial condition {name} must be finite")
    return result


def _snapshot(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.flags.writeable = False
    return result


def _require_finite_state(data: mujoco.MjData) -> None:
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise FloatingPointError("MuJoCo state became non-finite")
