"""Mechanics-first validation metrics for observations and model rollouts."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from . import dataset
from . import preprocessing
from . import simulation


MECHANICS_THRESHOLDS = {
    "minimum_countermovement_depth_m": 0.08,
    "preferred_countermovement_depth_low_m": 0.10,
    "preferred_countermovement_depth_high_m": 0.25,
    "minimum_no_contact_interval_s": 0.12,
    "maximum_impulse_flight_residual_s": 0.052,
    "minimum_flight_clearance_m": 0.001,
    "maximum_landing_rebound_m": 0.02,
    "maximum_posterior_drift_m": 0.08,
    "maximum_abs_torso_pitch_rad": 0.25,
    "maximum_abs_torso_pitch_rate_rad_s": 8.0,
    "maximum_lpt_tether_force_N": 5.0,
    "maximum_auxiliary_force_norm_N": 1e-9,
}


def _array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _rmse(observed: Any, predicted: Any) -> float:
    delta = _array(predicted) - _array(observed)
    return float(np.sqrt(np.mean(delta * delta)))


def _max_abs_error(observed: Any, predicted: Any) -> float:
    delta = np.abs(_array(predicted) - _array(observed))
    return float(np.max(delta)) if delta.size else 0.0


def _interpolate(predicted: dict[str, Any], key: str, target_time_s: Any) -> np.ndarray:
    return preprocessing.resample(predicted[key], predicted["time_s"], target_time_s)


def _observed_event_time(trial: dict[str, Any], key: str) -> float | None:
    value = trial.get("observed_events", {}).get(key)
    return None if value is None else float(value)


def _physical_metrics(result: dict[str, Any]) -> dict[str, Any]:
    traces = result["traces"]
    events = result["events"]
    time_s = _array(traces["time_s"])
    fz_total = _array(traces["fz_total_N"])
    root_z = _array(traces["root_z_m"])
    takeoff = events.get("takeoff_index")
    onset = int(events["movement_onset_index"])
    if takeoff is not None:
        takeoff = int(takeoff)
        depth = float(root_z[onset] - np.min(root_z[onset : takeoff + 1]))
    else:
        depth = 0.0
    contacts = np.asarray(traces["left_foot_contact"], dtype=bool) | np.asarray(
        traces["right_foot_contact"], dtype=bool
    )
    no_contact_duration = 0.0
    if takeoff is not None:
        landing = events.get("landing_index")
        end = int(landing) if landing is not None else len(contacts) - 1
        no_contact_duration = max(0.0, float(time_s[end] - time_s[takeoff]))
    finite = True
    for key, values in traces.items():
        if key in {"left_foot_contact", "right_foot_contact"} or values is None:
            continue
        # The source measurement path reports CoP as undefined outside
        # contact.  It therefore uses NaN as a physical sentinel rather than
        # as evidence of a failed rollout.  Preserve that source semantics,
        # while still rejecting infinities in the optional channel.
        if key == "cop_x_m":
            numeric = np.asarray(
                [float(value) for value in values if value is not None], dtype=float
            )
            if numeric.size and np.any(np.isinf(numeric)):
                finite = False
            continue
        try:
            numeric = _array(values)
        except (TypeError, ValueError):
            # Optional channels such as CoP contain None outside contact.
            numeric = np.asarray([float(value) for value in values if value is not None], dtype=float)
        if numeric.size and not np.all(np.isfinite(numeric)):
            finite = False
    metrics: dict[str, Any] = {
        "finite": bool(finite),
        "countermovement_depth_m": depth,
        "minimum_force_N": float(np.min(fz_total)),
        "maximum_force_N": float(np.max(fz_total)),
        "no_contact_duration_s": no_contact_duration,
        "takeoff_detected": takeoff is not None,
        "landing_detected": events.get("landing_index") is not None,
        "phase_order_valid": bool(events.get("phase_order_valid", False)),
        "any_contact_samples": int(np.count_nonzero(contacts)),
    }
    mechanics = _mechanics_validity(result)
    metrics["mechanics_metrics"] = mechanics["metrics"]
    metrics["mechanics_gates"] = mechanics["gates"]
    metrics["mechanics_valid"] = mechanics["valid"]
    return metrics


def _mechanics_validity(result: dict[str, Any]) -> dict[str, Any]:
    """Apply the source physical diagnostics without any aggregate grading."""

    traces = result["traces"]
    events = result["events"]
    summary = result["summary"]
    diagnostics = result.get("diagnostics", {})
    time_s = _array(traces["time_s"])
    root_z = _array(traces["root_z_m"])
    root_x = _array(traces["root_x_m"])
    force = _array(traces["fz_total_N"])
    clearance = _array(traces["foot_clearance_m"])
    pitch = _array(traces["root_pitch_rad"])
    pitch_rate = _array(traces["root_pitch_rate_rad_s"])
    lpt_force = _array(traces["lpt_tether_force_N"])
    phase_index = np.asarray(traces["phase_index"], dtype=int)
    left_contact = np.asarray(traces["left_foot_contact"], dtype=bool)
    right_contact = np.asarray(traces["right_foot_contact"], dtype=bool)
    takeoff_index = events.get("takeoff_index")
    landing_index = events.get("landing_index")
    if takeoff_index is None or landing_index is None:
        return {
            "valid": False,
            "metrics": {},
            "gates": {
                "used_mujoco": bool(result.get("used_mujoco")),
                "plant_valid": bool(result.get("valid")),
                "events_complete": False,
            },
        }
    takeoff_index = int(takeoff_index)
    landing_index = int(landing_index)
    onset_index = int(events["movement_onset_index"])
    if not (0 <= onset_index <= takeoff_index < landing_index < len(time_s)):
        return {"valid": False, "metrics": {}, "gates": {"events_ordered": False}}

    takeoff_velocity = float(summary["takeoff_velocity_m_s"])
    airborne_duration = float(summary["airborne_duration_s"])
    flight_residual = abs(2.0 * max(0.0, takeoff_velocity) / 9.81 - airborne_duration)
    depth = float(root_z[onset_index] - np.min(root_z[onset_index : takeoff_index + 1]))
    quiet_weight = max(float(summary["quiet_baseline_mean_N"]), 1e-9)
    min_fz_bw = float(np.min(force[onset_index : takeoff_index + 1]) / quiet_weight)
    landing_root_z = float(root_z[landing_index])
    landing_root_x = float(root_x[landing_index])
    landing_rebound = float(max(0.0, np.max(root_z[landing_index:]) - landing_root_z))
    posterior_drift = float(max(0.0, landing_root_x - np.min(root_x[landing_index:])))
    post_landing_contact = left_contact[landing_index:] | right_contact[landing_index:]
    no_rebound_flight = not any(
        not bool(post_landing_contact[i]) and not bool(post_landing_contact[i - 1])
        for i in range(1, len(post_landing_contact))
    )
    phase_count = len(diagnostics.get("phase_index_names", ("w",) * 7))
    mechanics_metrics = {
        "countermovement_depth_m": depth,
        "min_fz_bw_onset_to_takeoff": min_fz_bw,
        "measured_no_contact_interval_s": airborne_duration,
        "predicted_flight_time_s": 2.0 * max(0.0, takeoff_velocity) / 9.81,
        "impulse_flight_residual_s": flight_residual,
        "flight_clearance_max_m": float(np.max(clearance[takeoff_index : landing_index + 1])),
        "landing_rebound_m": landing_rebound,
        "posterior_drift_m": posterior_drift,
        "torso_pitch_abs_max_rad": float(np.max(np.abs(pitch))),
        "torso_pitch_rate_abs_max_rad_s": float(np.max(np.abs(pitch_rate))),
        "lpt_tether_force_abs_max_N": float(np.max(np.abs(lpt_force))),
        "auxiliary_force_norm_max_N": float(diagnostics.get("qfrc_applied_norm_max", math.inf)),
    }
    gates = {
        "used_mujoco": result.get("used_mujoco") is True and diagnostics.get("used_mujoco") is True,
        "plant_valid": result.get("valid") is True,
        "all_phases_observed": set(range(phase_count)).issubset(set(phase_index.tolist())),
        "countermovement_depth_ge_0_08_m": depth >= MECHANICS_THRESHOLDS["minimum_countermovement_depth_m"],
        "preferred_countermovement_depth_0_10_to_0_25_m": (
            MECHANICS_THRESHOLDS["preferred_countermovement_depth_low_m"]
            <= depth <= MECHANICS_THRESHOLDS["preferred_countermovement_depth_high_m"]
        ),
        "min_fz_bw_le_0_70": min_fz_bw <= 0.70,
        "measured_no_contact_ge_0_12_s": airborne_duration >= MECHANICS_THRESHOLDS["minimum_no_contact_interval_s"],
        "impulse_flight_residual_le_0_052_s": flight_residual <= MECHANICS_THRESHOLDS["maximum_impulse_flight_residual_s"],
        "positive_foot_clearance": mechanics_metrics["flight_clearance_max_m"] >= MECHANICS_THRESHOLDS["minimum_flight_clearance_m"],
        "touchdown_triggered_landing": bool(events.get("sustained_no_foot_contact")) and landing_index > takeoff_index,
        "no_rebound_mini_flight_after_landing": no_rebound_flight,
        "landing_rebound_le_0_02_m": landing_rebound <= MECHANICS_THRESHOLDS["maximum_landing_rebound_m"],
        "posterior_drift_le_0_08_m": posterior_drift <= MECHANICS_THRESHOLDS["maximum_posterior_drift_m"],
        "torso_pitch_bounded_abs_le_0_25_rad": mechanics_metrics["torso_pitch_abs_max_rad"] <= MECHANICS_THRESHOLDS["maximum_abs_torso_pitch_rad"],
        "torso_pitch_rate_bounded_abs_le_8_rad_s": mechanics_metrics["torso_pitch_rate_abs_max_rad_s"] <= MECHANICS_THRESHOLDS["maximum_abs_torso_pitch_rate_rad_s"],
        "lpt_force_bounded": mechanics_metrics["lpt_tether_force_abs_max_N"] <= MECHANICS_THRESHOLDS["maximum_lpt_tether_force_N"],
        "auxiliary_forces_zero": mechanics_metrics["auxiliary_force_norm_max_N"] <= MECHANICS_THRESHOLDS["maximum_auxiliary_force_norm_N"],
        "qpos_qvel_not_replayed_after_init": diagnostics.get("qpos_qvel_write_after_init") is False,
        "all_metrics_finite": all(math.isfinite(float(value)) for value in mechanics_metrics.values()),
    }
    return {"valid": all(gates.values()), "metrics": mechanics_metrics, "gates": gates}


def observation_residual_vector(
    trials: Iterable[dict[str, Any]],
    params: dict[str, Any],
    *,
    sample_count: int = 32,
) -> np.ndarray:
    """Return unit-balanced force/bar residuals for bounded fitting."""

    residuals: list[float] = []
    for trial in trials:
        result = simulation.simulate_trial(params, trial, record=True)
        observations = trial["observations"]
        time_s = _array(observations["time_s"])
        if sample_count > 0 and len(time_s) > sample_count:
            indices = np.linspace(0, len(time_s) - 1, sample_count, dtype=int)
        else:
            indices = np.arange(len(time_s))
        target_time = time_s[indices]
        predicted_fz = _interpolate(result["traces"], "fz_total_N", target_time)
        predicted_bar = _interpolate(result["traces"], "bar_displacement_m", target_time)
        observed_fz = _array(observations["fz_total_N"])[indices]
        observed_bar = _array(observations["bar_displacement_m"])[indices]
        force_scale = max(float(trial.get("observed_summary", {}).get("quiet_baseline_mean_N", 1.0)), 1.0)
        residuals.extend(((predicted_fz - observed_fz) / force_scale).tolist())
        residuals.extend(((predicted_bar - observed_bar) / 0.1).tolist())

        observed_summary = trial.get("observed_summary", {})
        predicted_summary = result["summary"]
        for key, scale in (
            ("takeoff_velocity_m_s", 1.0),
            ("jump_height_im_m", 0.1),
            ("propulsive_impulse_Ns", 100.0),
            ("bar_displacement_range_m", 0.1),
            ("bar_peak_velocity_m_s", 1.0),
        ):
            if key in observed_summary and predicted_summary.get(key) is not None:
                residuals.append((float(predicted_summary[key]) - float(observed_summary[key])) / scale)
    vector = np.asarray(residuals, dtype=float)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("nonfinite or empty identification residual vector")
    return vector


def validate_trial(
    params: dict[str, Any],
    trial: dict[str, Any],
) -> dict[str, Any]:
    """Compare one observed trial with a full-morphology forward rollout."""

    result = simulation.simulate_trial(params, trial, record=True)
    observations = trial["observations"]
    observed_time = _array(observations["time_s"])
    predicted_fz = _interpolate(result["traces"], "fz_total_N", observed_time)
    predicted_bar = _interpolate(result["traces"], "bar_displacement_m", observed_time)
    predicted_bar_velocity = _interpolate(result["traces"], "bar_velocity_m_s", observed_time)
    observed_fz = _array(observations["fz_total_N"])
    observed_bar = _array(observations["bar_displacement_m"])
    observed_bar_velocity = _array(observations.get("bar_velocity_m_s", np.gradient(observed_bar, observed_time)))

    trace_errors = {
        "force_plate_total_rmse_N": _rmse(observed_fz, predicted_fz),
        "force_plate_total_max_abs_error_N": _max_abs_error(observed_fz, predicted_fz),
        "bar_displacement_rmse_m": _rmse(observed_bar, predicted_bar),
        "bar_displacement_max_abs_error_m": _max_abs_error(observed_bar, predicted_bar),
        "bar_velocity_rmse_m_s": _rmse(observed_bar_velocity, predicted_bar_velocity),
    }
    observed_summary = trial.get("observed_summary", {})
    predicted_summary = result["summary"]
    summary_errors: dict[str, float] = {}
    for key in (
        "takeoff_velocity_m_s",
        "jump_height_im_m",
        "propulsive_impulse_Ns",
        "bar_displacement_range_m",
        "bar_peak_velocity_m_s",
    ):
        if key in observed_summary and predicted_summary.get(key) is not None:
            summary_errors[key] = abs(float(predicted_summary[key]) - float(observed_summary[key]))

    event_errors: dict[str, float | None] = {}
    for key in ("movement_onset_time_s", "takeoff_time_s", "landing_time_s"):
        observed = _observed_event_time(trial, key)
        predicted = result["events"].get(key)
        event_errors[key] = None if observed is None or predicted is None else abs(float(predicted) - observed)

    physical = _physical_metrics(result)
    valid = bool(
        result.get("valid", False)
        and physical["finite"]
        and physical["mechanics_valid"]
        and physical["takeoff_detected"]
        and physical["landing_detected"]
        and physical["phase_order_valid"]
    )
    return {
        "trial_id": trial.get("trial_id"),
        "valid": valid,
        "trace_errors": trace_errors,
        "summary_errors": summary_errors,
        "event_errors": event_errors,
        "physical": physical,
        "predicted_events": result["events"],
        "predicted_summary": predicted_summary,
        "result": result,
    }


def validate_parameters(
    params: dict[str, Any],
    trials: Iterable[dict[str, Any]] | None = None,
    *,
    max_trials: int | None = None,
) -> dict[str, Any]:
    """Run mechanics/measurement validation over supplied public trials."""

    active_trials = list(dataset.load_trials("validation") if trials is None else trials)
    if max_trials is not None:
        active_trials = active_trials[: int(max_trials)]
    reports: list[dict[str, Any]] = []
    for trial in active_trials:
        try:
            reports.append(validate_trial(params, trial))
        except Exception as exc:
            reports.append({
                "trial_id": trial.get("trial_id"),
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace_errors": {},
                "summary_errors": {},
                "event_errors": {},
                "physical": {},
            })
    force_errors = [r["trace_errors"]["force_plate_total_rmse_N"] for r in reports if r.get("trace_errors")]
    bar_errors = [r["trace_errors"]["bar_displacement_rmse_m"] for r in reports if r.get("trace_errors")]
    return {
        "valid": bool(reports) and all(bool(report["valid"]) for report in reports),
        "trial_count": len(reports),
        "trials": reports,
        "aggregate": {
            "mean_force_plate_total_rmse_N": float(np.mean(force_errors)) if force_errors else None,
            "max_force_plate_total_rmse_N": float(np.max(force_errors)) if force_errors else None,
            "mean_bar_displacement_rmse_m": float(np.mean(bar_errors)) if bar_errors else None,
            "max_bar_displacement_rmse_m": float(np.max(bar_errors)) if bar_errors else None,
        },
    }
