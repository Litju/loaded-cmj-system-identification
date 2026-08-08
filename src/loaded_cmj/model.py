"""MuJoCo model construction and compiled-plant integrity summaries."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from . import plant


def build_model(params: dict[str, Any] | None = None, trial: dict[str, Any] | None = None) -> mujoco.MjModel:
    """Compile the committed loaded-CMJ XML and apply scientific parameters."""

    return plant.build_model(plant.default_params() if params is None else params, trial)


def create_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Create a data object and apply the plant's deterministic initial state."""

    data = plant.reset_data(model)
    mujoco.mj_forward(model, data)
    return data


def _names(model: mujoco.MjModel, object_type: Any, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]


def compiled_model_signature(model: mujoco.MjModel) -> dict[str, Any]:
    """Return named counts/settings and critical arrays for equivalence audits."""

    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nsite": int(model.nsite),
        "nmesh": int(model.nmesh),
        "ntendon": int(model.ntendon),
        "nsensor": int(model.nsensor),
        "npair": int(model.npair),
        "neq": int(model.neq),
        "bodies": _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "joints": _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "geoms": _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        "sites": _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        "actuators": _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        "sensors": _names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
        "tendons": _names(model, mujoco.mjtObj.mjOBJ_TENDON, model.ntendon),
        "timestep_s": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
        "iterations": int(model.opt.iterations),
        "tolerance": float(model.opt.tolerance),
        "cone": int(model.opt.cone),
        "body_mass_kg": np.asarray(model.body_mass, dtype=float).tolist(),
        "body_inertia_kg_m2": np.asarray(model.body_inertia, dtype=float).tolist(),
        "jnt_stiffness": np.asarray(model.jnt_stiffness, dtype=float).tolist(),
        "dof_damping": np.asarray(model.dof_damping, dtype=float).tolist(),
        "geom_solref": np.asarray(model.geom_solref, dtype=float).tolist(),
        "pair_solref": np.asarray(model.pair_solref, dtype=float).tolist(),
        "tendon_stiffness": np.asarray(model.tendon_stiffness, dtype=float).tolist(),
        "tendon_damping": np.asarray(model.tendon_damping, dtype=float).tolist(),
    }

