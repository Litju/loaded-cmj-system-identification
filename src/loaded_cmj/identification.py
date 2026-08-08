"""Bounded system identification using the original MuJoCo forward model."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares

from . import dataset
from .parameters import default_parameters, flatten_parameters, unflatten_parameters
from .validation import observation_residual_vector


DEFAULT_FIT_COORDINATES = (
    "body_mass_kg",
    "joint_stiffness_Nm_rad[0]",
    "joint_stiffness_Nm_rad[1]",
    "joint_stiffness_Nm_rad[2]",
    "joint_damping_Nm_s_rad[0]",
    "joint_damping_Nm_s_rad[1]",
    "joint_damping_Nm_s_rad[2]",
    "braking_gain",
    "propulsive_gain",
    "bar_rack_stiffness_N_m",
    "bar_rack_damping_N_s_m",
    "encoder_scale",
    "encoder_delay_steps",
    "encoder_filter_tau_s",
    "encoder_offset_m",
)


def identify_parameters(
    trials: Iterable[dict[str, Any]] | None = None,
    *,
    initial: dict[str, Any] | None = None,
    parameter_names: Iterable[str] | None = None,
    max_nfev: int = 3,
    residual_sample_count: int = 32,
    verbose: int = 0,
) -> dict[str, Any]:
    """Fit bounded parameter coordinates against observed force/bar channels.

    The optimizer is intentionally explicit about the coordinates it varies.
    The returned configuration always contains the complete 27-coordinate
    structured parameter vector; omitted coordinates remain at ``initial``.
    """

    active_trials = list(dataset.load_trials("identification") if trials is None else trials)
    if not active_trials:
        raise ValueError("at least one identification trial is required")
    base = default_parameters() if initial is None else initial
    coordinates = tuple(parameter_names or DEFAULT_FIT_COORDINATES)
    names, x0, lower, upper = flatten_parameters(base, coordinates)
    rollout_failures = 0

    def residual(vector: np.ndarray) -> np.ndarray:
        nonlocal rollout_failures
        params = unflatten_parameters(base, names, vector)
        try:
            return observation_residual_vector(active_trials, params, sample_count=residual_sample_count)
        except (FloatingPointError, RuntimeError, ValueError, OverflowError):
            # A bounded trial that leaves the valid mechanics region is given a
            # deterministic finite penalty so the optimizer can move away from it.
            rollout_failures += 1
            return np.full(1, 1.0e6, dtype=float)

    fit = least_squares(
        residual,
        np.asarray(x0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=int(max_nfev),
        x_scale="jac",
        loss="linear",
        verbose=int(verbose),
    )
    fitted = unflatten_parameters(base, names, fit.x)
    residual_final = residual(np.asarray(fit.x, dtype=float))
    return {
        "parameters": fitted,
        "initial_parameters": base,
        "parameter_names": names,
        "success": bool(fit.success and rollout_failures == 0),
        "message": str(fit.message),
        "cost": float(fit.cost),
        "residual_norm": float(np.linalg.norm(residual_final)),
        "nfev": int(fit.nfev),
        "njev": None if fit.njev is None else int(fit.njev),
        "rollout_failures": int(rollout_failures),
    }
