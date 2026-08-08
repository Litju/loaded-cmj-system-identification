"""Public deterministic plant utilities for loaded CMJ force-plate/LPT sysid.

This module is intentionally public task infrastructure. It validates bounded
static parameter submissions, builds the locked linked sagittal MuJoCo model
when MuJoCo is available, and exposes deterministic trial, preprocessing, event,
and summary utilities for future public data generation and scoring slices.
The force plate signal is Fz_total from both feet. Takeoff requires sustained
no-foot-contact. LPT is bar-only, not COM. hIM is impulse-momentum jump height.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

try:  # Optional in plain import probes; required for MuJoCo runtime rollouts.
    import mujoco  # type: ignore
except Exception:  # pragma: no cover - exercised only without runtime deps.
    mujoco = None  # type: ignore

try:  # Optional in plain import probes; pure-Python fallbacks are kept below.
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - exercised only without runtime deps.
    np = None  # type: ignore


G = 9.81
DT = 0.002
SETTLE_DURATION_S = 2.0
CONTACT_FORCE_THRESHOLD_N = 20.0
MAX_QVEL_NORM = 200.0
MAX_ROOT_ABS_X_M = 2.0
MIN_ROOT_Z_M = -0.25
MAX_ROOT_Z_M = 2.5
MAX_ROOT_ABS_PITCH_RAD = 1.5
MAX_FZ_TOTAL_N = 45000.0
MAX_ABS_NET_IMPULSE_NS = 2000.0
MAX_LPT_IMPULSE_FRACTION = 0.10
ROOT_PITCH_STIFFNESS_NM_RAD = 3200.0
ROOT_PITCH_DAMPING_NM_S_RAD = 300.0
ROOT_PITCH_SPRINGREF_RAD = 0.0
# Sagittal anatomical contract for the plant:
# MuJoCo leg hinge axes are [0, -1, 0]; negative knee qpos/ctrl is flexion.
LEG_HINGE_AXIS = "0 -1 0"

# Frontier A1 (5E-B) XML-backed model integration.
# The committed assets/loaded_cmj_model.xml is the authoritative morphology
# (nq=nv=21, nu=6, three-body feet, six named contact geoms, passive
# forefoot-rocker/MTP joints). System-identification parameters are injected
# into a parsed copy of that model so morphology stays fixed while the plant is
# parameterizable; inertiafromgeom recomputes segment inertia from injected mass.
MODEL_XML_FILENAME = "loaded_cmj_model.xml"
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PACKAGE_DIR.parents[1]
_CONFIG_DIR = _REPOSITORY_ROOT / "configs"
_ASSET_DIR = _REPOSITORY_ROOT / "assets"
FOOT_CONTACT_GEOMS = (
    "left_heel",
    "left_forefoot",
    "left_toe",
    "right_heel",
    "right_forefoot",
    "right_toe",
)
LEG_ACTUATOR_ORDER = (
    "left_hip_torque",
    "left_knee_torque",
    "left_ankle_torque",
    "right_hip_torque",
    "right_knee_torque",
    "right_ankle_torque",
)
PASSIVE_FOOT_JOINTS = (
    "left_forefoot_rocker",
    "left_mtp",
    "right_forefoot_rocker",
    "right_mtp",
)
# The committed morphology's segment masses correspond to a 75 kg body and its
# bar to a 20 kg external load; parameterized masses scale from these references.
_NOMINAL_BODY_MASS_KG = 75.0
_NOMINAL_BAR_MASS_KG = 20.0


def _load_clean_core() -> Any:
    module_name = "_loaded_cmj_clean_core"
    module_path = Path(__file__).with_name("clean_core.py").resolve()
    if module_name in sys.modules:
        module = sys.modules[module_name]
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != module_path:
            raise ImportError("clean-core module is not bound to the public plant root")
        return module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load clean-core plant module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_clean_core = _load_clean_core()
# Frontier A1 six-torque controller (software PD position tracking + phase
# feedforward + trunk posture stabilization). The submitted joint stiffness/damping
# modulate the PD gains, but are scaled into a bounded, numerically stable band:
# joint_stiffness ~6500/8200/4800 maps to per-joint PD gains ~290/370/215 Nm/rad,
# which statically supports the loaded stance (~932 N over the six foot contacts)
# without the ankle PD limit cycle that raw stiffness magnitudes provoke at dt=2 ms.
# The bounds keep extreme submitted parameters from destabilizing the integrator
# while preserving parameter influence on the produced force trace for sysid.
# The schema spans a deliberately broad identification range.  A hard clamp
# made most of that range observationally identical, so normal values now use a
# monotone soft map into the stable controller envelope.  Only invalid values
# are rejected by schema validation; the tanh tails are reached only near the
# schema endpoints and never create a many-to-one normal-domain clamp.
_PD_KP_SAFE_LOW_NM_RAD = 300.0
_PD_KP_SAFE_HIGH_NM_RAD = 520.0
_PD_KD_SAFE_LOW_NM_S_RAD = 20.0
_PD_KD_SAFE_HIGH_NM_S_RAD = 80.0
_FF_TORQUE_SCALE = 0.22
# During PROPULSION the position-PD is softened to (kp_scale, kd_scale) so the
# smoothly ramped feedforward drives a flat, sustained concentric push (plausible
# ~3-4 x bodyweight force) rather than a stiff-PD spike; ctrlrange still caps torque.
_PROPULSION_PD_SCALE = (0.30, 0.65)
_POSTURE_KP_PITCH_NM_RAD = 320.0
_POSTURE_KD_PITCH_NM_S_RAD = 60.0
_POSTURE_KX_ROOT_NM_M = 40.0
_POSTURE_CAP_NM = 220.0
_ROOT_PITCH_QUIET_TARGET_RAD = 0.12
_ROOT_PITCH_HINGE_TARGET_RAD = 0.15
_ROOT_PITCH_TOEOFF_TARGET_RAD = 0.13
_ROOT_PITCH_LANDING_TARGET_RAD = 0.09
_UNWEIGHTING_POSE_RAMP_S = 0.32
_UNWEIGHTING_FF_SCALE = 0.65
_BRAKING_FF_RAMP_S = 0.36
_BRAKING_FF_SCALE = 0.90
_PROPULSION_FF_SCALE = 0.50
_PROPULSION_ACTIVATION_TAU_SCALE = 1.15
_CONTACT_FZ_EPS_N = 1e-6
# Reset height (root_z slide offset from the 1.00 m pelvis home) that seats all six
# foot contact geoms nearly flat on the plate with minimal initial penetration.
_RESET_ROOT_Z_M = -0.09
# New loaded-cmj-telemetry-v2 per-timestep channels exposed on the rollout trace
# in addition to (never replacing) the canonical scorer channels. Written as
# explicit literals (one per foot contact geom) so the exact channel names are
# discoverable in source, e.g. left_heel_fz_N / left_heel_fx_N.
_REGION_CONTACT_KEYS = (
    "left_heel_contact",
    "left_forefoot_contact",
    "left_toe_contact",
    "right_heel_contact",
    "right_forefoot_contact",
    "right_toe_contact",
)
_REGION_FZ_KEYS = (
    "left_heel_fz_N",
    "left_forefoot_fz_N",
    "left_toe_fz_N",
    "right_heel_fz_N",
    "right_forefoot_fz_N",
    "right_toe_fz_N",
)
_REGION_FX_KEYS = (
    "left_heel_fx_N",
    "left_forefoot_fx_N",
    "left_toe_fx_N",
    "right_heel_fx_N",
    "right_forefoot_fx_N",
    "right_toe_fx_N",
)
assert _REGION_CONTACT_KEYS == tuple(f"{g}_contact" for g in FOOT_CONTACT_GEOMS)
assert _REGION_FZ_KEYS == tuple(f"{g}_fz_N" for g in FOOT_CONTACT_GEOMS)
assert _REGION_FX_KEYS == tuple(f"{g}_fx_N" for g in FOOT_CONTACT_GEOMS)
NEW_TELEMETRY_TRACE_KEYS = (
    _REGION_CONTACT_KEYS
    + _REGION_FZ_KEYS
    + _REGION_FX_KEYS
    + (
        "total_fz_N",
        "total_fx_N",
        "cop_x_m",
        "left_forefoot_rocker_rad",
        "left_mtp_rad",
        "right_forefoot_rocker_rad",
        "right_mtp_rad",
        "left_slip_vx_m_s",
        "right_slip_vx_m_s",
    )
)
# Exact state channels used by the renderer's joint/COM telemetry panel.  These
# are read-only MuJoCo qpos/qvel/subtree-com values recorded during the same
# rollout; they are not new derived measurements or alternate plant outputs.
EXTENDED_KINEMATICS_TRACE_KEYS = (
    "com_x_m",
    "com_z_m",
    "root_x_velocity_m_s",
    "root_z_velocity_m_s",
    "lumbar_pitch_rad",
    "lumbar_pitch_rate_rad_s",
    "left_hip_velocity_rad_s",
    "right_hip_velocity_rad_s",
    "left_knee_velocity_rad_s",
    "right_knee_velocity_rad_s",
    "left_ankle_velocity_rad_s",
    "right_ankle_velocity_rad_s",
    "left_forefoot_rocker_velocity_rad_s",
    "right_forefoot_rocker_velocity_rad_s",
    "left_mtp_velocity_rad_s",
    "right_mtp_velocity_rad_s",
    "left_shoulder_rad",
    "right_shoulder_rad",
    "left_elbow_rad",
    "right_elbow_rad",
    "left_shoulder_velocity_rad_s",
    "right_shoulder_velocity_rad_s",
    "left_elbow_velocity_rad_s",
    "right_elbow_velocity_rad_s",
    "bar_rack_x_m",
    "bar_rack_z_m",
    "bar_rack_pitch_rad",
    "bar_rack_x_velocity_m_s",
    "bar_rack_z_velocity_m_s",
    "bar_rack_pitch_velocity_rad_s",
)
PHASES = {
    "weighing": (0.00, 1.50),
    "unweighting": (1.50, 1.82),
    "braking": (1.82, 2.12),
    "propulsion": (2.12, 2.58),
    "flight": (2.58, 2.95),
    "landing_absorption": (2.95, 3.25),
    "stabilization": (3.25, 3.60),
    "descent": (1.50, 1.82),
    "early_flight": (2.58, 2.95),
    "landing_diagnostic": (2.95, 3.60),
}
PHASE_INDEX = {
    "WEIGHING": 0,
    "UNWEIGHTING": 1,
    "BRAKING": 2,
    "PROPULSION": 3,
    "FLIGHT": 4,
    "LANDING_ABSORPTION": 5,
    "STABILIZATION": 6,
}
PHASE_NAMES = tuple(PHASE_INDEX.keys())
CANONICAL_TRACE_KEYS = (
    "time_s",
    "fz_left_N",
    "fz_right_N",
    "fz_total_N",
    "fnet_N",
    "root_z_m",
    "root_x_m",
    "root_pitch_rad",
    "root_pitch_rate_rad_s",
    "phase_index",
    "bar_z_m",
    "bar_displacement_m",
    "bar_velocity_m_s",
    "lpt_tether_force_N",
    "left_foot_contact",
    "right_foot_contact",
    "left_heel_z_m",
    "left_toe_z_m",
    "left_forefoot_z_m",
    "right_heel_z_m",
    "right_toe_z_m",
    "right_forefoot_z_m",
    "foot_clearance_m",
    "left_hip_rad",
    "left_knee_rad",
    "left_ankle_rad",
    "right_hip_rad",
    "right_knee_rad",
    "right_ankle_rad",
)
CANONICAL_SUMMARY_KEYS = (
    "quiet_baseline_mean_N",
    "quiet_baseline_sd_N",
    "quiet_baseline_cv",
    "movement_onset_time_s",
    "takeoff_time_s",
    "takeoff_velocity_m_s",
    "jump_height_im_m",
    "airborne_duration_s",
    "root_z_airborne_peak_gain_m",
    "propulsive_impulse_Ns",
    "total_net_impulse_Ns",
    "peak_concentric_force_N",
    "mean_concentric_force_N",
    "bar_peak_velocity_m_s",
    "bar_displacement_range_m",
    "max_lpt_tether_force_N",
    "lpt_tether_impulse_Ns",
    "lpt_tether_impulse_fraction_of_propulsive_impulse",
)

_DEFAULT_PARAMS = {
    "body_mass_kg": 74.2,
    "joint_stiffness_Nm_rad": [6100.0, 7850.0, 4550.0],
    "joint_damping_Nm_s_rad": [205.0, 245.0, 170.0],
    "braking_gain": 1.52,
    "propulsive_gain": 1.72,
    "activation_delay_s": 0.052,
    "activation_time_constant_s": 0.105,
    "bar_rack_stiffness_N_m": 25200.0,
    "bar_rack_damping_N_s_m": 780.0,
    "hand_grip_stiffness_N_m": 15800.0,
    "hand_grip_damping_N_s_m": 510.0,
    "contact_stiffness_N_m": 138000.0,
    "contact_damping_N_s_m": 5600.0,
    "force_plate_bias_N": -3.5,
    "force_plate_scale": 1.009,
    "encoder_scale": 0.991,
    "encoder_delay_steps": 0,
    "encoder_filter_tau_s": 0.006,
    "encoder_offset_m": -0.006,
    "bar_attachment_offset_m": [-0.014, 0.012],
    "lpt_tether_stiffness_N_m": 0.0,
    "lpt_tether_damping_N_s_m": 0.0,
}


def load_params(path: str | Path) -> dict[str, Any]:
    """Load and validate one submitted params JSON file."""

    try:
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in params file: {exc}") from exc
    if not isinstance(params, dict):
        raise ValueError("params JSON must be an object")
    validate_params(params)
    return copy.deepcopy(params)


def validate_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strictly validate finite values, exact keys, shapes, and schema bounds."""

    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    schema = _load_schema()
    properties = schema.get("properties", {})
    expected = list(schema.get("required", properties.keys()))
    expected_set = set(expected)
    actual_set = set(params.keys())
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        raise ValueError(f"missing required params: {missing}")
    if extra:
        raise ValueError(f"unexpected params are not allowed: {extra}")

    for excluded in schema.get("x-excluded-submitted-fields", []):
        if excluded in actual_set:
            raise ValueError(f"excluded submitted field present: {excluded}")

    for key in expected:
        spec = properties[key]
        _validate_value(key, params[key], spec)

    return params


def default_params() -> dict[str, Any]:
    """Return the public nominal seed params used for examples and smoke tests."""

    path = _CONFIG_DIR / "nominal.json"
    if path.exists():
        return load_params(path)
    params = copy.deepcopy(_DEFAULT_PARAMS)
    validate_params(params)
    return params


def build_model(params: dict[str, Any], trial: dict[str, Any] | None = None) -> Any:
    """Load the committed Frontier A1 morphology and apply sysid parameters.

    The bipedal sagittal loaded-CMJ morphology (nq=nv=21, nu=6, three-body feet,
    six named foot contact geoms with explicit force-plate pairs, passive
    forefoot-rocker/MTP joints, six hip/knee/ankle torque motors) is authored and
    frozen in assets/loaded_cmj_model.xml and loaded directly with
    mujoco.MjModel.from_xml_path. System-identification parameters are then applied
    in place to the compiled model (contact solref, bar-rack and tendon
    stiffness/damping, segment/bar mass and inertia). Morphology, topology, the
    actuator set, contact pairs, and sensors are never regenerated in code.
    """

    if mujoco is None:
        raise ImportError("mujoco is not available in this Python runtime")
    validate_params(params)
    trial = _default_trial(trial)
    configuration = _clean_core.ModelConfiguration(
        xml_path=_model_xml_path(),
        post_compile=lambda model: _apply_params_to_model(model, params, trial),
    )
    return _clean_core.build_model(configuration)


def reset_data(model: Any, data: Any | None = None) -> Any:
    """Reset MuJoCo data for a compiled model and return a finite initial state."""

    if mujoco is None:
        raise ImportError("mujoco is not available in this Python runtime")
    data = _clean_core.create_data(model) if data is None else data
    quiet_pose = _quiet_leg_pose()
    initial_condition = {
        "root_x": 0.0,
        "root_z": _RESET_ROOT_Z_M,
        "root_pitch": 0.0,
    }
    for side in ("left", "right"):
        for joint, value in zip(("hip", "knee", "ankle"), quiet_pose):
            initial_condition[f"{side}_{joint}"] = value
    _clean_core.reset_state(model, data, initial_condition)
    return data


def run_trial(
    params: dict[str, Any],
    trial: dict[str, Any] | None = None,
    record: bool = True,
) -> dict[str, Any]:
    """Run a deterministic public loaded-CMJ trial and return traces/summary."""

    validate_params(params)
    trial = _default_trial(trial)
    if mujoco is None:
        raise ImportError("mujoco is required for MuJoCo-primary rollouts")
    model = build_model(params, trial)
    data = reset_data(model)

    history = _simulate_trial_history(params, trial, model=model, data=data, record=record)
    traces = extract_traces(model, data, history, trial)
    for key in NEW_TELEMETRY_TRACE_KEYS:
        if key in history:
            traces[key] = history[key]
    for key in EXTENDED_KINEMATICS_TRACE_KEYS:
        if key in history:
            traces[key] = history[key]
    events = detect_events(traces)
    summary = summarize_trial(traces)
    diagnostics = history.get("diagnostics", {})
    diagnostics.update(_derive_landing_telemetry(traces, events))
    _validate_rollout_result(traces, events, summary, diagnostics)
    return {
        "summary": summary,
        "traces": traces,
        "trace": traces,
        "events": events,
        "used_mujoco": True,
        "valid": True,
        "diagnostics": diagnostics,
        "phase_timing_s": copy.deepcopy(PHASES),
    }


def _derive_landing_telemetry(
    traces: dict[str, Any], events: dict[str, Any]
) -> dict[str, Any]:
    """Landing/toe-off scalar telemetry (loaded-cmj-telemetry-v2 event channels).

    All values are derived read-only from the recorded trace and detected events;
    nothing is replayed. secondary_flight_count is the hard no-mini-hop signal.
    """

    time_s = _float_list(traces["time_s"])
    root_z = _float_list(traces["root_z_m"])
    left_c = _bool_list(traces["left_foot_contact"])
    right_c = _bool_list(traces["right_foot_contact"])
    takeoff_index = events.get("takeoff_index")
    landing_index = events.get("landing_index")
    region_fz = {
        g: _float_list(traces[f"{g}_fz_N"])
        for g in FOOT_CONTACT_GEOMS
        if f"{g}_fz_N" in traces
    }

    secondary_flight_count = 0
    landing_rebound_m = 0.0
    if landing_index is not None:
        li = int(landing_index)
        in_gap = False
        for i in range(li + 1, len(time_s)):
            airborne = (not left_c[i]) and (not right_c[i])
            if airborne and not in_gap:
                in_gap = True
                secondary_flight_count += 1
            elif not airborne:
                in_gap = False
        post = root_z[li:]
        if post:
            landing_rebound_m = max(0.0, max(post) - root_z[li])

    def _region_of(geom: str) -> str:
        if "heel" in geom:
            return "heel"
        return "forefoot" if "forefoot" in geom else "toe"

    touchdown_first_region = "none"
    if takeoff_index is not None:
        for i in range(int(takeoff_index), len(time_s)):
            hits = {
                _region_of(g)
                for g, series in region_fz.items()
                if series[i] > CONTACT_FORCE_THRESHOLD_N
            }
            if hits:
                touchdown_first_region = hits.pop() if len(hits) == 1 else "mixed"
                break

    heel_off_time_s: float | None = None
    if takeoff_index is not None:
        heel_series = [
            region_fz.get("left_heel", [0.0] * len(time_s))[i]
            + region_fz.get("right_heel", [0.0] * len(time_s))[i]
            for i in range(len(time_s))
        ]
        for i in range(int(takeoff_index), 0, -1):
            if heel_series[i - 1] > CONTACT_FORCE_THRESHOLD_N:
                heel_off_time_s = time_s[i]
                break

    return {
        "secondary_flight_count": secondary_flight_count,
        "landing_rebound_m": landing_rebound_m,
        "touchdown_first_region": touchdown_first_region,
        "heel_off_time_s": heel_off_time_s,
    }


def extract_traces(
    model: Any,
    data: Any,
    history: dict[str, Any],
    trial: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract canonical force, state, bar, LPT, and contact traces."""

    del model, data, trial
    if isinstance(history.get("traces"), dict):
        history = history["traces"]
    elif isinstance(history.get("trace"), dict):
        history = history["trace"]

    trace: dict[str, Any] = {}
    for key in CANONICAL_TRACE_KEYS:
        if key in history:
            trace[key] = _maybe_array(history[key])

    time_s = _float_list(trace["time_s"])
    if "fz_total_N" not in trace:
        left = _float_list(trace["fz_left_N"])
        right = _float_list(trace["fz_right_N"])
        trace["fz_total_N"] = _maybe_array([a + b for a, b in zip(left, right)])
    if "fnet_N" not in trace:
        processed = preprocess_force_trace(trace)
        trace["fnet_N"] = processed["fnet_N"]

    lengths = {len(_sequence(trace[key])) for key in CANONICAL_TRACE_KEYS if key in trace}
    if len(lengths) != 1 or next(iter(lengths)) != len(time_s):
        raise ValueError("canonical traces must have equal length")
    for key in CANONICAL_TRACE_KEYS:
        if key not in trace:
            raise ValueError(f"missing canonical trace key: {key}")
        values = _sequence(trace[key])
        if key.endswith("_contact"):
            continue
        for value in values:
            if not math.isfinite(float(value)):
                raise ValueError(f"nonfinite trace value in {key}")
    return trace


def preprocess_force_trace(
    trace: dict[str, Any],
    preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply public weighing, total/net force, mass, and event-prep rules."""

    cfg = _default_preprocessing(preprocessing)
    out = dict(trace)
    time_s = _float_list(out["time_s"])
    if not time_s or time_s[-1] < cfg["quiet_duration_s"]:
        raise ValueError("trace must include quiet weighing duration >= 1.50 s")

    if "fz_total_N" in out:
        fz_total = _float_list(out["fz_total_N"])
    else:
        left = _float_list(out["fz_left_N"])
        right = _float_list(out["fz_right_N"])
        fz_total = [a + b for a, b in zip(left, right)]
        out["fz_total_N"] = _maybe_array(fz_total)

    quiet_end = cfg["quiet_duration_s"]
    quiet_start = max(0.0, quiet_end - cfg["baseline_window_s"])
    baseline_idx = [
        i for i, t in enumerate(time_s) if quiet_start <= t <= quiet_end
    ]
    if not baseline_idx:
        raise ValueError("no samples in final stable 1.00 s weighing window")

    baseline = [fz_total[i] for i in baseline_idx]
    wsys = _mean(baseline)
    quiet_sd = _std(baseline)
    if not math.isfinite(wsys) or wsys <= 0.0:
        raise ValueError("invalid quiet baseline system weight")

    fnet = [fz - wsys for fz in fz_total]
    msys = wsys / G
    out["fz_total_N"] = _maybe_array(fz_total)
    out["fnet_N"] = _maybe_array(fnet)
    out["Wsys_N"] = wsys
    out["msys_kg"] = msys
    out["quiet_force_sd_N"] = quiet_sd
    out["quiet_baseline_mean_N"] = wsys
    out["quiet_baseline_sd_N"] = quiet_sd
    out["quiet_baseline_cv"] = quiet_sd / abs(wsys)
    out["aCOM_m_s2"] = _maybe_array([x / msys for x in fnet])
    return out


def detect_events(
    trace: dict[str, Any],
    preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect public movement onset, takeoff, landing, and phase validity."""

    tr = preprocess_force_trace(trace, preprocessing)
    cfg = _default_preprocessing(preprocessing)
    time_s = _float_list(tr["time_s"])
    fz_total = _float_list(tr["fz_total_N"])
    fnet = _float_list(tr["fnet_N"])
    bar_disp = _float_list(tr["bar_displacement_m"])
    bar_vel = _float_list(tr["bar_velocity_m_s"])
    left_contact = _bool_list(tr["left_foot_contact"])
    right_contact = _bool_list(tr["right_foot_contact"])
    wsys = float(tr["Wsys_N"])
    msys = float(tr["msys_kg"])
    quiet_sd = float(tr["quiet_force_sd_N"])
    dt = _median_dt(time_s)

    onset_threshold_N = wsys - max(5.0 * quiet_sd, 20.0)
    onset_sustain_n = max(1, int(math.ceil(cfg["movement_sustain_s"] / dt)))
    onset_idx = _first_sustained_index(
        [fz < onset_threshold_N for fz in fz_total],
        time_s,
        start_time=cfg["quiet_duration_s"],
        sustain_n=onset_sustain_n,
    )
    quiet_bar_idx = [
        i for i, t in enumerate(time_s)
        if cfg["quiet_duration_s"] - cfg["baseline_window_s"] <= t <= cfg["quiet_duration_s"]
    ]
    quiet_bar_mean = _mean([bar_disp[i] for i in quiet_bar_idx])
    quiet_bar_sd = _std([bar_disp[i] for i in quiet_bar_idx])
    bar_disp_threshold = max(0.004, 5.0 * quiet_bar_sd)
    bar_vel_threshold = 0.020
    bar_onset_idx = _first_sustained_index(
        [
            abs(value - quiet_bar_mean) > bar_disp_threshold
            or abs(velocity) > bar_vel_threshold
            for value, velocity in zip(bar_disp, bar_vel)
        ],
        time_s,
        start_time=cfg["quiet_duration_s"],
        sustain_n=onset_sustain_n,
    )
    if bar_onset_idx is not None:
        onset_idx = min(onset_idx, bar_onset_idx) if onset_idx is not None else bar_onset_idx
    if onset_idx is None:
        onset_idx = _first_index_after(time_s, cfg["quiet_duration_s"])

    no_contact = [
        (not bool(lc)) and (not bool(rc))
        for lc, rc in zip(left_contact, right_contact)
    ]
    takeoff_sustain_n = max(1, int(math.ceil(cfg["takeoff_sustain_s"] / dt)))
    detected_takeoff_idx = _first_sustained_index(
        no_contact,
        time_s,
        start_time=PHASES["propulsion"][0],
        sustain_n=takeoff_sustain_n,
    )
    sustained_no_foot_contact = detected_takeoff_idx is not None
    takeoff_idx = detected_takeoff_idx
    if takeoff_idx is None:
        takeoff_idx = _first_index_after(time_s, PHASES["propulsion"][1])

    recontact = [
        bool(lc) or bool(rc) for lc, rc in zip(left_contact, right_contact)
    ]
    landing_idx = _first_sustained_index(
        recontact,
        time_s,
        start_time=time_s[takeoff_idx] + cfg["takeoff_sustain_s"],
        sustain_n=takeoff_sustain_n,
    )

    takeoff_velocity = _integrate(time_s, [x / msys for x in fnet], onset_idx, takeoff_idx)
    takeoff_time = time_s[takeoff_idx]
    landing_time = time_s[landing_idx] if landing_idx is not None else None
    airborne_duration = (
        max(0.0, landing_time - takeoff_time) if landing_time is not None else 0.0
    )

    return {
        "quiet_duration_s": cfg["quiet_duration_s"],
        "quiet_baseline_window_s": cfg["baseline_window_s"],
        "movement_onset_threshold_N": onset_threshold_N,
        "movement_onset_index": onset_idx,
        "movement_onset_time_s": time_s[onset_idx],
        "takeoff_index": takeoff_idx,
        "takeoff_time_s": takeoff_time,
        "landing_index": landing_idx,
        "landing_time_s": landing_time,
        "takeoff_velocity_m_s": takeoff_velocity,
        "jump_height_im_m": takeoff_velocity * takeoff_velocity / (2.0 * G),
        "airborne_duration_s": airborne_duration,
        "phase_order_valid": time_s[onset_idx] >= PHASES["weighing"][1]
        and takeoff_time >= PHASES["propulsion"][0],
        "sustained_no_foot_contact": sustained_no_foot_contact,
    }


def summarize_trial(trace: dict[str, Any]) -> dict[str, float]:
    """Return canonical scalar mechanics and bar/LPT diagnostic metrics."""

    tr = preprocess_force_trace(trace)
    events = detect_events(tr)
    time_s = _float_list(tr["time_s"])
    fz_total = _float_list(tr["fz_total_N"])
    fnet = _float_list(tr["fnet_N"])
    root_z = _float_list(tr["root_z_m"])
    bar_disp = _float_list(tr["bar_displacement_m"])
    bar_vel = _float_list(tr["bar_velocity_m_s"])
    lpt_force = _float_list(tr["lpt_tether_force_N"])

    onset_idx = int(events["movement_onset_index"])
    takeoff_idx = int(events["takeoff_index"])
    landing_idx = events.get("landing_index")
    # Propulsive (concentric) window is the measured phase from the countermovement
    # bottom (deepest root_z, where vertical velocity reverses) to takeoff, not a
    # fixed scripted PHASES constant. With foot articulation the feet leave the
    # ground at the propulsion-phase boundary, so a constant window collapses to an
    # empty interval; the measured concentric window yields a physically meaningful,
    # nonzero propulsive impulse (~ msys * takeoff_velocity).
    if takeoff_idx > onset_idx:
        prop_start_idx = min(range(onset_idx, takeoff_idx + 1), key=lambda i: root_z[i])
    else:
        prop_start_idx = onset_idx
    prop_end_idx = max(prop_start_idx, takeoff_idx)

    propulsive_impulse = _integrate(time_s, fnet, prop_start_idx, prop_end_idx)
    total_net_impulse = _integrate(time_s, fnet, onset_idx, takeoff_idx)
    prop_slice = fz_total[prop_start_idx : prop_end_idx + 1] or [0.0]
    movement_slice = bar_disp[onset_idx : takeoff_idx + 1] or [0.0]
    lpt_impulse = _integrate(time_s, lpt_force, prop_start_idx, prop_end_idx)

    if landing_idx is None:
        air_root = root_z[takeoff_idx:]
    else:
        air_root = root_z[takeoff_idx : int(landing_idx) + 1]
    root_peak_gain = max(air_root) - root_z[takeoff_idx] if air_root else 0.0

    summary = {
        "quiet_baseline_mean_N": float(tr["quiet_baseline_mean_N"]),
        "quiet_baseline_sd_N": float(tr["quiet_baseline_sd_N"]),
        "quiet_baseline_cv": float(tr["quiet_baseline_cv"]),
        "movement_onset_time_s": float(events["movement_onset_time_s"]),
        "takeoff_time_s": float(events["takeoff_time_s"]),
        "takeoff_velocity_m_s": float(events["takeoff_velocity_m_s"]),
        "jump_height_im_m": float(events["jump_height_im_m"]),
        "airborne_duration_s": float(events["airborne_duration_s"]),
        "root_z_airborne_peak_gain_m": float(root_peak_gain),
        "propulsive_impulse_Ns": float(propulsive_impulse),
        "total_net_impulse_Ns": float(total_net_impulse),
        "peak_concentric_force_N": float(max(prop_slice)),
        "mean_concentric_force_N": float(_mean(prop_slice)),
        "bar_peak_velocity_m_s": float(max(bar_vel)),
        "bar_displacement_range_m": float(max(movement_slice) - min(movement_slice)),
        "max_lpt_tether_force_N": float(max(lpt_force) if lpt_force else 0.0),
        "lpt_tether_impulse_Ns": float(lpt_impulse),
        "lpt_tether_impulse_fraction_of_propulsive_impulse": float(
            abs(lpt_impulse) / max(abs(propulsive_impulse), 1e-12)
        ),
    }
    for key in CANONICAL_SUMMARY_KEYS:
        if key not in summary:
            raise ValueError(f"missing canonical summary key: {key}")
        if not math.isfinite(summary[key]):
            raise ValueError(f"nonfinite summary value for {key}")
    return summary


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is not allowed: {value}")


def _load_schema() -> dict[str, Any]:
    path = _CONFIG_DIR / "param_schema.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_value(name: str, value: Any, spec: dict[str, Any]) -> None:
    typ = spec.get("type")
    if typ == "number":
        if not _is_plain_number(value):
            raise ValueError(f"{name} must be a finite number")
        v = float(value)
        if v < float(spec["minimum"]) or v > float(spec["maximum"]):
            raise ValueError(f"{name} out of bounds")
        return
    if typ == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < int(spec["minimum"]) or value > int(spec["maximum"]):
            raise ValueError(f"{name} out of bounds")
        return
    if typ == "array":
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array")
        min_items = int(spec.get("minItems", spec.get("length", 0)))
        max_items = int(spec.get("maxItems", spec.get("length", min_items)))
        if len(value) < min_items or len(value) > max_items:
            raise ValueError(f"{name} must have length {min_items}")
        if "prefixItems" in spec:
            for i, item_spec in enumerate(spec["prefixItems"]):
                _validate_value(f"{name}[{i}]", value[i], item_spec)
        else:
            item_spec = spec["items"]
            for i, item in enumerate(value):
                _validate_value(f"{name}[{i}]", item, item_spec)
        return
    raise ValueError(f"unsupported schema type for {name}: {typ}")


def _is_plain_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if np is not None:
        try:
            if isinstance(value, np.generic):
                value = value.item()
        except Exception:
            pass
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _default_trial(trial: dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "external_load_kg": 20.0,
        "depth_scale": 1.0,
        "braking_duration_scale": 1.0,
        "propulsion_duration_scale": 1.0,
        "force_plate_noise_sd_N": 0.0,
        "duration_s": PHASES["landing_diagnostic"][1],
        "dt_s": DT,
    }
    if trial:
        defaults.update(trial)
    return defaults


def _default_preprocessing(preprocessing: dict[str, Any] | None) -> dict[str, float]:
    cfg = {
        "quiet_duration_s": 1.50,
        "baseline_window_s": 1.00,
        "movement_sustain_s": 0.050,
        "takeoff_sustain_s": 0.025,
    }
    if preprocessing:
        cfg.update({k: float(v) for k, v in preprocessing.items()})
    if cfg["quiet_duration_s"] < 1.50:
        raise ValueError("quiet weighing duration must be >= 1.50 s")
    return cfg


def _model_xml_path() -> Path:
    """Absolute path to the committed Frontier A1 morphology XML."""

    return _ASSET_DIR / MODEL_XML_FILENAME


def _apply_params_to_model(model: Any, params: dict[str, Any], trial: dict[str, Any]) -> None:
    """Apply system-identification parameters in place to a compiled model.

    Only parameter-dependent scalars are mutated on the already-compiled Frontier
    A1 model: foot-contact and force-plate solref (from contact stiffness/damping),
    explicit force-plate pair solref, bar-rack joint stiffness/damping, hand-grip
    and LPT tendon stiffness/damping, and segment/bar mass with inertia scaled
    proportionally (mass scales linearly for fixed geometry, so inertia scales by
    the same factor). Morphology, topology, actuator set, joints, contact pairs and
    sensors are untouched; measurement-model parameters (force-plate bias/scale,
    encoder response, bar attachment offset, LPT gains) stay in the Python
    measurement model and are not part of the physical model.
    """

    body_mass = float(params["body_mass_kg"])
    bar_mass = max(5.0, float(trial.get("external_load_kg", 20.0)))
    contact_k = float(params["contact_stiffness_N_m"])
    contact_d = float(params["contact_damping_N_s_m"])
    system_mass = max(body_mass + bar_mass, 1.0)
    contact_timeconst = _clamp(
        math.sqrt(system_mass / max(contact_k, 1.0)), 0.005, 0.080
    )
    contact_dampratio = _clamp(
        contact_d / (2.0 * math.sqrt(max(contact_k * system_mass, 1.0))),
        0.70,
        2.00,
    )

    for name in FOOT_CONTACT_GEOMS + ("force_plate",):
        geom_id = _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_solref[geom_id] = [contact_timeconst, contact_dampratio]
    for pair_index in range(int(model.npair)):
        model.pair_solref[pair_index] = [contact_timeconst, contact_dampratio]

    joint_stiffness_damping = {
        "bar_rack_x": (params["bar_rack_stiffness_N_m"], params["bar_rack_damping_N_s_m"]),
        "bar_rack_z": (params["bar_rack_stiffness_N_m"], params["bar_rack_damping_N_s_m"]),
    }
    for joint_name, (stiffness, damping) in joint_stiffness_damping.items():
        joint_id = _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        model.jnt_stiffness[joint_id] = float(stiffness)
        model.dof_damping[model.jnt_dofadr[joint_id]] = float(damping)

    tendon_stiffness_damping = {
        "left_hand_grip": (params["hand_grip_stiffness_N_m"], params["hand_grip_damping_N_s_m"]),
        "right_hand_grip": (params["hand_grip_stiffness_N_m"], params["hand_grip_damping_N_s_m"]),
        "lpt_tether": (params["lpt_tether_stiffness_N_m"], params["lpt_tether_damping_N_s_m"]),
    }
    for tendon_name, (stiffness, damping) in tendon_stiffness_damping.items():
        tendon_id = _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name)
        model.tendon_stiffness[tendon_id] = float(stiffness)
        model.tendon_damping[tendon_id] = float(damping)

    # The committed model masses correspond to a 75 kg body and a 20 kg bar, so the
    # segment bodies scale uniformly by body_mass/75 and the bar by external_load/20.
    bar_body_id = _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, "bar")
    segment_factor = body_mass / _NOMINAL_BODY_MASS_KG
    bar_factor = bar_mass / _NOMINAL_BAR_MASS_KG
    for body_id in range(int(model.nbody)):
        if body_id == bar_body_id:
            factor = bar_factor
        elif float(model.body_mass[body_id]) > 0.0:
            factor = segment_factor
        else:
            continue
        model.body_mass[body_id] *= factor
        model.body_inertia[body_id] *= factor


def _leg_actuator_index(model: Any) -> dict[str, int]:
    """Deterministic actuator-name -> ctrl index map for the six leg torque motors.

    Built by name lookup, not positional assumption, so control assignment is
    correct regardless of actuator declaration order. Enforces the nu==6 contract.
    """

    if int(model.nu) != len(LEG_ACTUATOR_ORDER):
        raise ValueError(
            f"expected nu={len(LEG_ACTUATOR_ORDER)} torque actuators, got nu={int(model.nu)}"
        )
    index: dict[str, int] = {}
    for name in LEG_ACTUATOR_ORDER:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"missing MuJoCo actuator: {name}")
        index[name] = int(actuator_id)
    return index


def _leg_pd_ctrl(
    model: Any,
    data: Any,
    act_index: dict[str, int],
    params: dict[str, Any],
    pose: tuple[float, float, float],
    ff_torque: tuple[float, float, float],
    root_pitch: float = 0.0,
    root_pitch_rate: float = 0.0,
    root_x: float = 0.0,
    root_pitch_target: float = 0.0,
    pd_scale: tuple[float, float] = (1.0, 1.0),
) -> list[float]:
    """Frontier A1 six-torque control law -> ctrl vector of length model.nu.

    Each hip/knee/ankle motor applies a software PD position-tracking torque toward
    the phase pose target plus a scaled phase feedforward torque, clamped to the
    actuator ctrlrange. The submitted joint stiffness/damping act as the PD gains so
    system-identification parameters shape the produced force trace. The hips also
    carry a bounded trunk-posture stabilization term. During the concentric
    propulsion phase the position-PD is softened (pd_scale) so the smoothly ramped
    feedforward drives a flat, sustained triple-extension push instead of a single
    stiff-PD force spike, while the actuator ctrlrange still caps every motor. Passive
    forefoot-rocker/MTP joints are never actuated (no motor exists for them). No
    qpos/qvel is written and no external force is applied; only data.ctrl is set.
    """

    kp_scale, kd_scale = pd_scale
    kp = [
        _soft_schema_map(float(x), 1000.0, 30000.0, _PD_KP_SAFE_LOW_NM_RAD, _PD_KP_SAFE_HIGH_NM_RAD)
        * kp_scale
        for x in params["joint_stiffness_Nm_rad"]
    ]
    kd = [
        _soft_schema_map(float(x), 10.0, 2000.0, _PD_KD_SAFE_LOW_NM_S_RAD, _PD_KD_SAFE_HIGH_NM_S_RAD)
        * kd_scale
        for x in params["joint_damping_Nm_s_rad"]
    ]
    posture = _clamp(
        -(
            _POSTURE_KP_PITCH_NM_RAD * (root_pitch - root_pitch_target)
            + _POSTURE_KD_PITCH_NM_S_RAD * root_pitch_rate
            + _POSTURE_KX_ROOT_NM_M * root_x
        ),
        -_POSTURE_CAP_NM,
        _POSTURE_CAP_NM,
    )
    ctrl = [0.0] * int(model.nu)
    for side in ("left", "right"):
        for joint_i, joint in enumerate(("hip", "knee", "ankle")):
            idx = act_index[f"{side}_{joint}_torque"]
            q = _qpos_value(model, data, f"{side}_{joint}")
            qd = _qvel_value(model, data, f"{side}_{joint}")
            tau = (
                kp[joint_i] * (pose[joint_i] - q)
                - kd[joint_i] * qd
                + _FF_TORQUE_SCALE * ff_torque[joint_i]
            )
            if joint_i == 0:  # hips carry trunk posture stabilization
                tau += posture
            lo = float(model.actuator_ctrlrange[idx][0])
            hi = float(model.actuator_ctrlrange[idx][1])
            ctrl[idx] = _clamp(tau, lo, hi)
    return ctrl


def _simulate_trial_history(
    params: dict[str, Any],
    trial: dict[str, Any],
    model: Any = None,
    data: Any = None,
    record: bool = True,
) -> dict[str, Any]:
    if mujoco is None:
        raise ImportError("mujoco is required for MuJoCo-primary rollouts")
    if model is None:
        model = build_model(params, trial)
    if data is None:
        data = reset_data(model)

    duration = float(trial.get("duration_s", PHASES["landing_diagnostic"][1]))
    dt = float(model.opt.timestep)
    n = int(round(duration / dt)) + 1
    time_s = [i * dt for i in range(n)]
    ids = _mujoco_measurement_ids(model)
    act_index = _leg_actuator_index(model)

    quiet_pose = _quiet_leg_pose()
    for _ in range(int(round(SETTLE_DURATION_S / dt))):
        _clean_core.apply_control(model, data, _leg_pd_ctrl(
            model,
            data,
            act_index,
            params,
            quiet_pose,
            (0.0, 0.0, 0.0),
            root_pitch_target=_ROOT_PITCH_QUIET_TARGET_RAD,
        ))
        _clean_core.step(model, data)
        _assert_finite_mujoco_state(data)
    data.time = 0.0

    fz_left: list[float] = []
    fz_right: list[float] = []
    root_z: list[float] = []
    root_x: list[float] = []
    root_pitch: list[float] = []
    bar_z: list[float] = []
    bar_site_positions: list[tuple[float, float, float]] = []
    anchor_positions: list[tuple[float, float, float]] = []
    root_pitch_rate: list[float] = []
    phase_index: list[int] = []
    left_heel_z: list[float] = []
    left_toe_z: list[float] = []
    left_forefoot_z: list[float] = []
    right_heel_z: list[float] = []
    right_toe_z: list[float] = []
    right_forefoot_z: list[float] = []
    foot_clearance: list[float] = []
    left_hip: list[float] = []
    left_knee: list[float] = []
    left_ankle: list[float] = []
    right_hip: list[float] = []
    right_knee: list[float] = []
    right_ankle: list[float] = []
    com_x: list[float] = []
    com_z: list[float] = []
    root_x_velocity: list[float] = []
    root_z_velocity: list[float] = []
    lumbar_pitch: list[float] = []
    lumbar_pitch_rate: list[float] = []
    left_hip_velocity: list[float] = []
    right_hip_velocity: list[float] = []
    left_knee_velocity: list[float] = []
    right_knee_velocity: list[float] = []
    left_ankle_velocity: list[float] = []
    right_ankle_velocity: list[float] = []
    left_rocker_velocity: list[float] = []
    right_rocker_velocity: list[float] = []
    left_mtp_velocity: list[float] = []
    right_mtp_velocity: list[float] = []
    left_shoulder: list[float] = []
    right_shoulder: list[float] = []
    left_elbow: list[float] = []
    right_elbow: list[float] = []
    left_shoulder_velocity: list[float] = []
    right_shoulder_velocity: list[float] = []
    left_elbow_velocity: list[float] = []
    right_elbow_velocity: list[float] = []
    bar_rack_x: list[float] = []
    bar_rack_z: list[float] = []
    bar_rack_pitch: list[float] = []
    bar_rack_x_velocity: list[float] = []
    bar_rack_z_velocity: list[float] = []
    bar_rack_pitch_velocity: list[float] = []
    qvel_norms: list[float] = []
    motor_torque_norms: list[float] = []
    aux_force_norms: list[float] = []
    xfrc_force_norms: list[float] = []
    region_contact: dict[str, list[bool]] = {g: [] for g in FOOT_CONTACT_GEOMS}
    region_fz: dict[str, list[float]] = {g: [] for g in FOOT_CONTACT_GEOMS}
    region_fx: dict[str, list[float]] = {g: [] for g in FOOT_CONTACT_GEOMS}
    total_fz_series: list[float] = []
    total_fx_series: list[float] = []
    cop_x_series: list[float | None] = []
    left_slip_series: list[float] = []
    right_slip_series: list[float] = []
    left_rocker: list[float] = []
    left_mtp: list[float] = []
    right_rocker: list[float] = []
    right_mtp: list[float] = []

    controller = _PhaseSupervisor(params, trial)
    prev_root_z = float(data.xpos[ids["pelvis_body"]][2])
    prev_geom_x = {g: float(data.geom_xpos[ids[f"geom::{g}"]][0]) for g in FOOT_CONTACT_GEOMS}

    for t in time_s:
        left_raw_pre, right_raw_pre, _, _ = _contact_forces(model, data, ids)
        support_force_pre = left_raw_pre + right_raw_pre
        current_root_z = float(data.xpos[ids["pelvis_body"]][2])
        current_root_x = float(data.xpos[ids["pelvis_body"]][0])
        current_pitch = _qpos_value(model, data, "root_pitch")
        current_pitch_rate = _qvel_value(model, data, "root_pitch")
        current_vz = (current_root_z - prev_root_z) / dt if t > 0.0 else 0.0
        pose, torque, phase = controller.control(
            t=t,
            root_z=current_root_z,
            root_x=current_root_x,
            root_vz=current_vz,
            root_pitch=current_pitch,
            support_force_N=support_force_pre,
        )
        motor_ctrl = _leg_pd_ctrl(
            model,
            data,
            act_index,
            params,
            pose,
            torque,
            root_pitch=current_pitch,
            root_pitch_rate=current_pitch_rate,
            root_x=current_root_x,
            root_pitch_target=controller.root_pitch_target(),
            pd_scale=_PROPULSION_PD_SCALE if phase == "PROPULSION" else (1.0, 1.0),
        )
        _clean_core.apply_control(model, data, motor_ctrl)
        _clean_core.step(model, data)
        _assert_finite_mujoco_state(data)
        prev_root_z = current_root_z

        left_raw, right_raw, per_fz, per_fx = _contact_forces(model, data, ids)
        left_measured, right_measured = _force_plate_measurement(
            left_raw, right_raw, params
        )
        fz_left.append(left_measured)
        fz_right.append(right_measured)

        # Per-region contact telemetry over the six named foot geoms.
        weighted_x = 0.0
        total_region_fz = 0.0
        for g in FOOT_CONTACT_GEOMS:
            fz_g = per_fz[g]
            fx_g = per_fx[g]
            region_fz[g].append(fz_g)
            region_fx[g].append(fx_g)
            in_contact = fz_g > CONTACT_FORCE_THRESHOLD_N
            region_contact[g].append(in_contact)
            geom_x = float(data.geom_xpos[ids[f"geom::{g}"]][0])
            weighted_x += fz_g * geom_x
            total_region_fz += fz_g
        total_fz_series.append(left_raw + right_raw)
        total_fx_series.append(sum(per_fx[g] for g in FOOT_CONTACT_GEOMS))
        cop_x_series.append(
            (weighted_x / total_region_fz) if total_region_fz > _CONTACT_FZ_EPS_N else None
        )
        for side, store in (("left", left_slip_series), ("right", right_slip_series)):
            slip = 0.0
            for g in FOOT_CONTACT_GEOMS:
                if not g.startswith(side):
                    continue
                geom_x = float(data.geom_xpos[ids[f"geom::{g}"]][0])
                if per_fz[g] > CONTACT_FORCE_THRESHOLD_N:
                    slip = max(slip, abs(geom_x - prev_geom_x[g]) / dt)
                prev_geom_x[g] = geom_x
            store.append(slip)
        left_rocker.append(_qpos_value(model, data, "left_forefoot_rocker"))
        left_mtp.append(_qpos_value(model, data, "left_mtp"))
        right_rocker.append(_qpos_value(model, data, "right_forefoot_rocker"))
        right_mtp.append(_qpos_value(model, data, "right_mtp"))

        root_x.append(float(data.xpos[ids["pelvis_body"]][0]))
        root_z.append(float(data.xpos[ids["pelvis_body"]][2]))
        root_pitch.append(_qpos_value(model, data, "root_pitch"))
        root_pitch_rate.append(_qvel_value(model, data, "root_pitch"))
        subtree_com = data.subtree_com[ids["pelvis_body"]]
        com_x.append(float(subtree_com[0]))
        com_z.append(float(subtree_com[2]))
        root_x_velocity.append(_qvel_value(model, data, "root_x"))
        root_z_velocity.append(_qvel_value(model, data, "root_z"))
        lumbar_pitch.append(_qpos_value(model, data, "lumbar_pitch"))
        lumbar_pitch_rate.append(_qvel_value(model, data, "lumbar_pitch"))
        phase_index.append(PHASE_INDEX[phase])

        bar_pos = _bar_measurement_position(model, data, ids, params)
        anchor_pos = tuple(float(x) for x in data.site_xpos[ids["lpt_anchor_site"]])
        bar_z.append(bar_pos[2])
        bar_site_positions.append(bar_pos)
        anchor_positions.append(anchor_pos)
        lhz = float(data.site_xpos[ids["left_heel_site"]][2])
        ltz = float(data.site_xpos[ids["left_toe_site"]][2])
        lfz = float(data.site_xpos[ids["left_forefoot_site"]][2])
        rhz = float(data.site_xpos[ids["right_heel_site"]][2])
        rtz = float(data.site_xpos[ids["right_toe_site"]][2])
        rfz = float(data.site_xpos[ids["right_forefoot_site"]][2])
        left_heel_z.append(lhz)
        left_toe_z.append(ltz)
        left_forefoot_z.append(lfz)
        right_heel_z.append(rhz)
        right_toe_z.append(rtz)
        right_forefoot_z.append(rfz)
        foot_clearance.append(min(lhz, ltz, lfz, rhz, rtz, rfz))
        left_hip.append(_qpos_value(model, data, "left_hip"))
        left_knee.append(_qpos_value(model, data, "left_knee"))
        left_ankle.append(_qpos_value(model, data, "left_ankle"))
        right_hip.append(_qpos_value(model, data, "right_hip"))
        right_knee.append(_qpos_value(model, data, "right_knee"))
        right_ankle.append(_qpos_value(model, data, "right_ankle"))
        left_hip_velocity.append(_qvel_value(model, data, "left_hip"))
        right_hip_velocity.append(_qvel_value(model, data, "right_hip"))
        left_knee_velocity.append(_qvel_value(model, data, "left_knee"))
        right_knee_velocity.append(_qvel_value(model, data, "right_knee"))
        left_ankle_velocity.append(_qvel_value(model, data, "left_ankle"))
        right_ankle_velocity.append(_qvel_value(model, data, "right_ankle"))
        left_rocker_velocity.append(_qvel_value(model, data, "left_forefoot_rocker"))
        right_rocker_velocity.append(_qvel_value(model, data, "right_forefoot_rocker"))
        left_mtp_velocity.append(_qvel_value(model, data, "left_mtp"))
        right_mtp_velocity.append(_qvel_value(model, data, "right_mtp"))
        left_shoulder.append(_qpos_value(model, data, "left_shoulder"))
        right_shoulder.append(_qpos_value(model, data, "right_shoulder"))
        left_elbow.append(_qpos_value(model, data, "left_elbow"))
        right_elbow.append(_qpos_value(model, data, "right_elbow"))
        left_shoulder_velocity.append(_qvel_value(model, data, "left_shoulder"))
        right_shoulder_velocity.append(_qvel_value(model, data, "right_shoulder"))
        left_elbow_velocity.append(_qvel_value(model, data, "left_elbow"))
        right_elbow_velocity.append(_qvel_value(model, data, "right_elbow"))
        bar_rack_x.append(_qpos_value(model, data, "bar_rack_x"))
        bar_rack_z.append(_qpos_value(model, data, "bar_rack_z"))
        bar_rack_pitch.append(_qpos_value(model, data, "bar_rack_pitch"))
        bar_rack_x_velocity.append(_qvel_value(model, data, "bar_rack_x"))
        bar_rack_z_velocity.append(_qvel_value(model, data, "bar_rack_z"))
        bar_rack_pitch_velocity.append(_qvel_value(model, data, "bar_rack_pitch"))
        qvel_norms.append(_vector_norm(data.qvel))
        motor_torque_norms.append(_vector_norm(motor_ctrl))
        aux_force_norms.append(_vector_norm(data.qfrc_applied) + _vector_norm(data.xfrc_applied))
        xfrc_force_norms.append(_vector_norm(data.xfrc_applied))

    fz_total = [left + right for left, right in zip(fz_left, fz_right)]
    force_threshold = CONTACT_FORCE_THRESHOLD_N
    left_contact, right_contact = _debounced_force_plate_contacts(
        time_s, fz_left, fz_right, force_threshold
    )

    quiet_bar_idx = [i for i, t in enumerate(time_s) if 0.50 <= t <= PHASES["weighing"][1]]
    if not quiet_bar_idx:
        raise ValueError("no quiet bar samples in MuJoCo rollout")
    quiet_bar = _mean([bar_z[i] for i in quiet_bar_idx])
    raw_bar_disp = [z - quiet_bar for z in bar_z]
    filtered_bar_disp = _first_order_filter(
        raw_bar_disp, time_s, float(params["encoder_filter_tau_s"])
    )
    delayed_bar_disp = _delay(filtered_bar_disp, int(params["encoder_delay_steps"]))
    bar_disp = [
        float(params["encoder_scale"]) * value + float(params["encoder_offset_m"])
        for value in delayed_bar_disp
    ]
    bar_vel = _differentiate(time_s, bar_disp)

    lengths = [
        math.sqrt(
            (bar[0] - anchor[0]) ** 2
            + (bar[1] - anchor[1]) ** 2
            + (bar[2] - anchor[2]) ** 2
        )
        for bar, anchor in zip(bar_site_positions, anchor_positions)
    ]
    rest_length = _mean([lengths[i] for i in quiet_bar_idx])
    length_rate = _differentiate(time_s, lengths)
    quiet_force = _mean([fz_total[i] for i in quiet_bar_idx])
    lpt_cap = min(10.0, 0.01 * max(quiet_force, 0.0))
    lpt_force = [
        _clamp(
            float(params["lpt_tether_stiffness_N_m"]) * max(0.0, length - rest_length)
            + float(params["lpt_tether_damping_N_s_m"]) * max(0.0, rate),
            0.0,
            lpt_cap,
        )
        for length, rate in zip(lengths, length_rate)
    ]

    trace = {
        "time_s": time_s,
        "fz_left_N": fz_left,
        "fz_right_N": fz_right,
        "fz_total_N": fz_total,
        "root_z_m": root_z,
        "root_x_m": root_x,
        "root_pitch_rad": root_pitch,
        "root_pitch_rate_rad_s": root_pitch_rate,
        "phase_index": phase_index,
        "bar_z_m": bar_z,
        "bar_displacement_m": bar_disp,
        "bar_velocity_m_s": bar_vel,
        "lpt_tether_force_N": lpt_force,
        "left_foot_contact": left_contact,
        "right_foot_contact": right_contact,
        "left_heel_z_m": left_heel_z,
        "left_toe_z_m": left_toe_z,
        "left_forefoot_z_m": left_forefoot_z,
        "right_heel_z_m": right_heel_z,
        "right_toe_z_m": right_toe_z,
        "right_forefoot_z_m": right_forefoot_z,
        "foot_clearance_m": foot_clearance,
        "left_hip_rad": left_hip,
        "left_knee_rad": left_knee,
        "left_ankle_rad": left_ankle,
        "right_hip_rad": right_hip,
        "right_knee_rad": right_knee,
        "right_ankle_rad": right_ankle,
        "com_x_m": com_x,
        "com_z_m": com_z,
        "root_x_velocity_m_s": root_x_velocity,
        "root_z_velocity_m_s": root_z_velocity,
        "lumbar_pitch_rad": lumbar_pitch,
        "lumbar_pitch_rate_rad_s": lumbar_pitch_rate,
        "left_hip_velocity_rad_s": left_hip_velocity,
        "right_hip_velocity_rad_s": right_hip_velocity,
        "left_knee_velocity_rad_s": left_knee_velocity,
        "right_knee_velocity_rad_s": right_knee_velocity,
        "left_ankle_velocity_rad_s": left_ankle_velocity,
        "right_ankle_velocity_rad_s": right_ankle_velocity,
        "left_forefoot_rocker_velocity_rad_s": left_rocker_velocity,
        "right_forefoot_rocker_velocity_rad_s": right_rocker_velocity,
        "left_mtp_velocity_rad_s": left_mtp_velocity,
        "right_mtp_velocity_rad_s": right_mtp_velocity,
        "left_shoulder_rad": left_shoulder,
        "right_shoulder_rad": right_shoulder,
        "left_elbow_rad": left_elbow,
        "right_elbow_rad": right_elbow,
        "left_shoulder_velocity_rad_s": left_shoulder_velocity,
        "right_shoulder_velocity_rad_s": right_shoulder_velocity,
        "left_elbow_velocity_rad_s": left_elbow_velocity,
        "right_elbow_velocity_rad_s": right_elbow_velocity,
        "bar_rack_x_m": bar_rack_x,
        "bar_rack_z_m": bar_rack_z,
        "bar_rack_pitch_rad": bar_rack_pitch,
        "bar_rack_x_velocity_m_s": bar_rack_x_velocity,
        "bar_rack_z_velocity_m_s": bar_rack_z_velocity,
        "bar_rack_pitch_velocity_rad_s": bar_rack_pitch_velocity,
    }
    for g in FOOT_CONTACT_GEOMS:
        trace[f"{g}_contact"] = region_contact[g]
        trace[f"{g}_fz_N"] = region_fz[g]
        trace[f"{g}_fx_N"] = region_fx[g]
    trace["total_fz_N"] = total_fz_series
    trace["total_fx_N"] = total_fx_series
    trace["cop_x_m"] = cop_x_series
    trace["left_forefoot_rocker_rad"] = left_rocker
    trace["left_mtp_rad"] = left_mtp
    trace["right_forefoot_rocker_rad"] = right_rocker
    trace["right_mtp_rad"] = right_mtp
    trace["left_slip_vx_m_s"] = left_slip_series
    trace["right_slip_vx_m_s"] = right_slip_series

    processed = preprocess_force_trace(trace)
    trace["fnet_N"] = _float_list(processed["fnet_N"])

    diagnostics = {
        "used_mujoco": True,
        "settle_duration_s": SETTLE_DURATION_S,
        "scored_step_count": n,
        "dt_s": dt,
        "qvel_norm_max": max(qvel_norms) if qvel_norms else 0.0,
        "root_x_abs_max_m": max(abs(x) for x in root_x) if root_x else 0.0,
        "root_z_min_m": min(root_z) if root_z else 0.0,
        "root_z_max_m": max(root_z) if root_z else 0.0,
        "root_pitch_abs_max_rad": max(abs(x) for x in root_pitch) if root_pitch else 0.0,
        "root_pitch_rate_abs_max_rad_s": max(abs(x) for x in root_pitch_rate) if root_pitch_rate else 0.0,
        "fz_total_peak_N": max(fz_total) if fz_total else 0.0,
        "phase_index_names": PHASE_NAMES,
        "observed_phase_indices": sorted(set(phase_index)),
        "motor_torque_norm_max_Nm": max(motor_torque_norms) if motor_torque_norms else 0.0,
        "qfrc_applied_norm_max": max(aux_force_norms) if aux_force_norms else 0.0,
        "xfrc_applied_norm_max": max(xfrc_force_norms) if xfrc_force_norms else 0.0,
        "qpos_qvel_write_after_init": False,
        "force_plate_contact_threshold_N": force_threshold,
        "contact_force_source": "mujoco.mj_contactForce",
        "contact_geoms": list(FOOT_CONTACT_GEOMS),
        "control_law": "frontier_a1_six_torque_pd_posture_feedforward",
        "control_vector_length": int(model.nu),
        "model_source": MODEL_XML_FILENAME,
        "final_joint_velocity_norm": _vector_norm(
            [
                _qvel_value(model, data, f"{side}_{joint}")
                for side in ("left", "right")
                for joint in ("hip", "knee", "ankle")
            ]
        ),
        "final_com_z_m": float(data.subtree_com[ids["pelvis_body"]][2]),
        "bar_source": "mujoco body/site positions",
        "lpt_source": "mujoco site positions measurement model",
    }

    if not record:
        keep = [0, _first_index_after(time_s, PHASES["descent"][0]), _first_index_after(time_s, PHASES["propulsion"][1]), len(time_s) - 1]
        trace = {key: [values[i] for i in keep] for key, values in trace.items()}
        diagnostics["record_downsampled"] = True

    trace["diagnostics"] = diagnostics
    return trace


def _quiet_leg_pose() -> tuple[float, float, float]:
    return -0.02, -0.09, 0.02


_THIGH_LENGTH_M = 0.34
_SHANK_LENGTH_M = 0.36
# The IK anchor is the loaded contact patch, not the ankle joint center.  The
# quiet-stance ankle FK underestimates the loaded heel/forefoot/toe centroid by
# about 5 cm in this three-body foot, so this offset keeps the modeled contact
# geoms invariant during squat and push-off without constraining root_x.
_FOOT_IK_ANCHOR_X_OFFSET_M = 0.048


def _sagittal_ankle_fk(hip: float, knee: float) -> tuple[float, float]:
    return (
        _THIGH_LENGTH_M * math.sin(hip) + _SHANK_LENGTH_M * math.sin(hip + knee),
        -_THIGH_LENGTH_M * math.cos(hip) - _SHANK_LENGTH_M * math.cos(hip + knee),
    )


def _solve_sagittal_leg_ik(ankle_x: float, ankle_z: float) -> tuple[float, float]:
    reach = _clamp(
        math.hypot(ankle_x, ankle_z),
        abs(_THIGH_LENGTH_M - _SHANK_LENGTH_M) + 1e-3,
        _THIGH_LENGTH_M + _SHANK_LENGTH_M - 1e-3,
    )
    phi = math.atan2(ankle_x, -ankle_z)
    knee_interior = math.acos(
        _clamp(
            (_THIGH_LENGTH_M * _THIGH_LENGTH_M + _SHANK_LENGTH_M * _SHANK_LENGTH_M - reach * reach)
            / (2.0 * _THIGH_LENGTH_M * _SHANK_LENGTH_M),
            -1.0,
            1.0,
        )
    )
    hip_offset = math.acos(
        _clamp(
            (_THIGH_LENGTH_M * _THIGH_LENGTH_M + reach * reach - _SHANK_LENGTH_M * _SHANK_LENGTH_M)
            / (2.0 * _THIGH_LENGTH_M * reach),
            -1.0,
            1.0,
        )
    )
    return phi + hip_offset, -(math.pi - knee_interior)


def _foot_invariant_pose(
    original_hip: float,
    original_knee: float,
    ankle: float,
) -> tuple[float, float, float]:
    anchor_x, _ = _sagittal_ankle_fk(*_quiet_leg_pose()[:2])
    anchor_x += _FOOT_IK_ANCHOR_X_OFFSET_M
    _, target_z = _sagittal_ankle_fk(original_hip, original_knee)
    hip, knee = _solve_sagittal_leg_ik(anchor_x, target_z)
    return (
        _clamp(hip, -0.90, 1.26),
        _clamp(knee, -1.85, 0.20),
        ankle,
    )


class _PhaseSupervisor:
    def __init__(self, params: dict[str, Any], trial: dict[str, Any]) -> None:
        self.params = params
        self.trial = trial
        self.phase = "WEIGHING"
        self.phase_start_s = 0.0
        self.quiet_forces: list[float] = []
        self.standing_z: float | None = None
        self.bottom_z: float | None = None
        self.takeoff_s: float | None = None
        self.touchdown_s: float | None = None
        self.no_contact_s = 0.0
        self.contact_s = 0.0
        self.prev_supported = True
        self.elapsed_s = 0.0

    def control(
        self,
        t: float,
        root_z: float,
        root_x: float,
        root_vz: float,
        root_pitch: float,
        support_force_N: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
        self.elapsed_s = max(0.0, t - self.phase_start_s)
        bw = self._body_weight_estimate()
        supported = support_force_N >= CONTACT_FORCE_THRESHOLD_N
        if supported:
            self.contact_s += DT
            self.no_contact_s = 0.0
        else:
            self.no_contact_s += DT
            self.contact_s = 0.0

        if self.phase == "WEIGHING":
            if support_force_N > CONTACT_FORCE_THRESHOLD_N:
                self.quiet_forces.append(support_force_N)
                if len(self.quiet_forces) > int(1.0 / DT):
                    self.quiet_forces.pop(0)
            if self.standing_z is None and t >= 0.50:
                self.standing_z = root_z
            if t >= PHASES["weighing"][1] and abs(root_vz) < 0.05:
                self._set_phase("UNWEIGHTING", t)

        depth = max(0.0, (self.standing_z or root_z) - root_z)
        fz_over_bw = support_force_N / max(bw, 1.0)

        if self.phase == "UNWEIGHTING":
            if (supported and fz_over_bw <= 0.70 and depth >= 0.050) or t - self.phase_start_s > self._scaled_duration("unweighting", 0.38):
                self._set_phase("BRAKING", t)
        elif self.phase == "BRAKING":
            braking_elapsed = t - self.phase_start_s
            if (
                depth >= 0.190
                and root_vz >= -0.04
                and braking_elapsed >= self._scaled_duration("braking", 0.22)
            ) or braking_elapsed > self._scaled_duration("braking", 0.60):
                self.bottom_z = root_z
                self._set_phase("PROPULSION", t)
        elif self.phase == "PROPULSION":
            if self.no_contact_s >= 0.018:
                self.takeoff_s = t
                self._set_phase("FLIGHT", t)
            elif t - self.phase_start_s > self._scaled_duration("propulsion", 0.52):
                self.takeoff_s = t
                self._set_phase("FLIGHT", t)
        elif self.phase == "FLIGHT":
            if supported and self.contact_s >= 0.010:
                self.touchdown_s = t
                self._set_phase("LANDING_ABSORPTION", t)
        elif self.phase == "LANDING_ABSORPTION":
            if (
                self.contact_s >= 0.16
                and abs(root_vz) < 0.10
                and abs(root_pitch - _ROOT_PITCH_QUIET_TARGET_RAD) < 0.07
            ):
                self._set_phase("STABILIZATION", t)
            elif t - self.phase_start_s > 0.42:
                self._set_phase("STABILIZATION", t)

        pose = self._pose(depth, root_vz)
        torque = self._torque(root_x, root_vz, root_pitch)
        self.prev_supported = supported
        return pose, torque, self.phase

    def root_pitch_target(self) -> float:
        if self.phase == "UNWEIGHTING":
            x = _smoothstep(_phase_x(self._phase_elapsed(), 0.0, self._scaled_duration("unweighting", 0.24)))
            return _ROOT_PITCH_QUIET_TARGET_RAD + (
                _ROOT_PITCH_HINGE_TARGET_RAD - _ROOT_PITCH_QUIET_TARGET_RAD
            ) * x
        if self.phase == "BRAKING":
            return _ROOT_PITCH_HINGE_TARGET_RAD
        if self.phase == "PROPULSION":
            x = _smoothstep(_phase_x(self._phase_elapsed(), 0.0, self._scaled_duration("propulsion", 0.16)))
            return _ROOT_PITCH_HINGE_TARGET_RAD + (
                _ROOT_PITCH_TOEOFF_TARGET_RAD - _ROOT_PITCH_HINGE_TARGET_RAD
            ) * x
        if self.phase == "FLIGHT":
            return _ROOT_PITCH_TOEOFF_TARGET_RAD
        if self.phase == "LANDING_ABSORPTION":
            return _ROOT_PITCH_LANDING_TARGET_RAD
        if self.phase == "STABILIZATION":
            x = _smoothstep(_phase_x(self._phase_elapsed(), 0.03, 0.30))
            return _ROOT_PITCH_LANDING_TARGET_RAD + (
                _ROOT_PITCH_QUIET_TARGET_RAD - _ROOT_PITCH_LANDING_TARGET_RAD
            ) * x
        return _ROOT_PITCH_QUIET_TARGET_RAD

    def _set_phase(self, phase: str, t: float) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_start_s = t
            self.elapsed_s = 0.0

    def _body_weight_estimate(self) -> float:
        if self.quiet_forces:
            return _mean(self.quiet_forces)
        mass = float(self.params["body_mass_kg"]) + float(self.trial.get("external_load_kg", 20.0))
        return mass * G

    def _pose(self, depth: float, root_vz: float) -> tuple[float, float, float]:
        del root_vz
        quiet = _quiet_leg_pose()
        depth_scale = _clamp(float(self.trial.get("depth_scale", 1.0)), 0.75, 1.35)
        braking_gain = _clamp(float(self.params["braking_gain"]), 0.2, 3.0)
        prop_gain = _clamp(float(self.params["propulsive_gain"]), 0.2, 3.5)
        deep = _foot_invariant_pose(
            1.10 + 0.06 * (braking_gain - 1.0) * depth_scale,
            -1.68 - 0.05 * (braking_gain - 1.0) * depth_scale,
            _clamp(0.44 + 0.04 * depth_scale, -0.55, 0.60),
        )
        # Ankle sign convention (verified against the compiled model): positive
        # ankle qpos is dorsiflexion (toe up), negative is plantarflexion (toe
        # down / heel up). Push-off targets ankle plantarflexion so the loaded CMJ
        # rocks onto the forefoot/toe and produces a real plantarflexion toe-off
        # (heel-off before forefoot/toe last-contact), not a heel-driven takeoff.
        extension = _foot_invariant_pose(
            -0.20 - 0.02 * (prop_gain - 1.0),
            0.18,
            _clamp(-0.38, -0.65, 0.55),
        )
        braking = _foot_invariant_pose(1.06, -1.62, _clamp(0.40, -0.55, 0.60))
        landing = (
            _clamp(0.32 + 0.04 * depth_scale, -0.90, 0.60),
            _clamp(-0.85, -1.20, 0.20),
            _clamp(0.16, -0.55, 0.55),
        )
        # Flight foot is plantarflexed (toe down) so touchdown contacts the
        # forefoot/toe first; the swing leg is tucked (knee flexed) so it does not
        # reach down and re-contact early, keeping landing height near takeoff height
        # (symmetric ballistic flight -> impulse/flight-time consistency).
        flight_recovery = (
            _clamp(0.58, -0.90, 0.60),
            _clamp(-1.00, -1.20, 0.20),
            _clamp(0.16, -0.55, 0.55),
        )
        if self.phase == "WEIGHING":
            return quiet
        if self.phase == "UNWEIGHTING":
            x = _smoothstep(_phase_x(self._phase_elapsed(), 0.0, _UNWEIGHTING_POSE_RAMP_S))
            return _interpolate_pose(quiet, deep, x)
        if self.phase == "BRAKING":
            hold = _clamp(depth / 0.20, 0.0, 1.0)
            return _interpolate_pose(deep, braking, 0.25 * hold)
        if self.phase == "PROPULSION":
            x = _smoothstep(_phase_x(self._phase_elapsed(), 0.0, 0.12))
            return _interpolate_pose(deep, extension, x)
        if self.phase == "FLIGHT":
            return flight_recovery
        if self.phase == "LANDING_ABSORPTION":
            return landing
        x = _smoothstep(_phase_x(self._phase_elapsed(), 0.03, 0.30))
        return _interpolate_pose(landing, quiet, x)

    def _torque(self, root_x: float, root_vz: float, root_pitch: float) -> tuple[float, float, float]:
        prop = _clamp(float(self.params["propulsive_gain"]), 0.2, 3.5)
        brake = _clamp(float(self.params["braking_gain"]), 0.2, 3.0)
        pitch_damp = _clamp(90.0 * (root_pitch - self.root_pitch_target()) + 16.0 * root_x, -45.0, 45.0)
        vz_damp = _clamp(35.0 * root_vz, -80.0, 80.0)
        if self.phase == "UNWEIGHTING":
            x = _smoothstep(_phase_x(self._phase_elapsed(), 0.0, self._scaled_duration("unweighting", 0.20)))
            return (
                80.0 * brake * x * _UNWEIGHTING_FF_SCALE - pitch_damp,
                -130.0 * brake * x * _UNWEIGHTING_FF_SCALE,
                35.0 * x * _UNWEIGHTING_FF_SCALE,
            )
        if self.phase == "BRAKING":
            x = _smoothstep(_phase_x(self._phase_elapsed(), 0.0, self._scaled_duration("braking", _BRAKING_FF_RAMP_S)))
            return (
                -220.0 * brake * x * _BRAKING_FF_SCALE - pitch_damp,
                220.0 * brake * x * _BRAKING_FF_SCALE - 0.4 * vz_damp,
                100.0 * brake * x * _BRAKING_FF_SCALE,
            )
        if self.phase == "PROPULSION":
            activation_delay = max(0.0, float(self.params["activation_delay_s"]))
            tau = max(
                float(self.params["activation_time_constant_s"]) * _PROPULSION_ACTIVATION_TAU_SCALE,
                0.015,
            )
            active_elapsed = max(0.0, self._phase_elapsed() - activation_delay)
            x = 1.0 - math.exp(-active_elapsed / tau)
            scale = prop * x * _PROPULSION_FF_SCALE
            # Ankle feedforward is a plantarflexion (negative) torque so the push
            # drives the ball of the foot into the plate for the toe-off.
            return (-590.0 * scale - pitch_damp, 1600.0 * scale, -1120.0 * scale)
        if self.phase == "LANDING_ABSORPTION":
            x = 1.0 - math.exp(-max(0.0, self._phase_elapsed()) / 0.08)
            return (40.0 * x - pitch_damp - 0.35 * vz_damp, 0.8 * vz_damp, 0.35 * vz_damp)
        if self.phase == "STABILIZATION":
            return (-pitch_damp, -0.35 * vz_damp, 0.0)
        return (0.0, 0.0, 0.0)

    def _phase_elapsed(self) -> float:
        return self.elapsed_s

    def _scaled_duration(self, phase: str, nominal_s: float) -> float:
        key = f"{phase}_duration_scale"
        scale = _clamp(float(self.trial.get(key, 1.0)), 0.75, 1.35)
        return nominal_s * scale


def _interpolate_pose(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    a = _clamp(alpha, 0.0, 1.0)
    return tuple(s + (e - s) * a for s, e in zip(start, end))  # type: ignore[return-value]


def _mujoco_measurement_ids(model: Any) -> dict[str, int]:
    ids = {
        "pelvis_body": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
        "bar_body": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_BODY, "bar"),
        "force_plate_geom": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, "force_plate"),
        "bar_lpt_site": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "bar_lpt_site"),
        "lpt_anchor_site": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "lpt_anchor"),
        "left_heel_site": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_heel_site"),
        "left_toe_site": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_toe_site"),
        "left_forefoot_site": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_forefoot_site"),
        "right_heel_site": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_heel_site"),
        "right_toe_site": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_toe_site"),
        "right_forefoot_site": _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_forefoot_site"),
    }
    for name in FOOT_CONTACT_GEOMS:
        ids[f"geom::{name}"] = _required_mujoco_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return ids


def _required_mujoco_id(model: Any, obj_type: Any, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"missing MuJoCo object: {name}")
    return int(idx)


def _contact_forces(
    model: Any, data: Any, ids: dict[str, int]
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    """Aggregate force-plate contact forces over the six named foot geoms.

    Returns (left_fz_total, right_fz_total, per_geom_fz, per_geom_fx) in world
    coordinates. Only force_plate<->foot-geom contacts are summed; the wrench is
    read from mujoco.mj_contactForce and rotated into the world frame. Vertical
    magnitude is accumulated per side; per-geom vertical and anteroposterior
    components feed the loaded-cmj-telemetry-v2 channels (CoP, region GRF, slip).
    """

    per_fz = {g: 0.0 for g in FOOT_CONTACT_GEOMS}
    per_fx = {g: 0.0 for g in FOOT_CONTACT_GEOMS}
    plate = ids["force_plate_geom"]
    geom_to_name = {ids[f"geom::{g}"]: g for g in FOOT_CONTACT_GEOMS}
    wrench = np.zeros(6) if np is not None else [0.0] * 6
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if g1 == plate and g2 in geom_to_name:
            name = geom_to_name[g2]
        elif g2 == plate and g1 in geom_to_name:
            name = geom_to_name[g1]
        else:
            continue
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        if np is not None:
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            world_force = frame.T @ np.asarray(wrench[:3], dtype=float)
            per_fz[name] += abs(float(world_force[2]))
            per_fx[name] += float(world_force[0])
        else:
            per_fz[name] += abs(float(wrench[2]))
    left = sum(per_fz[g] for g in FOOT_CONTACT_GEOMS if g.startswith("left"))
    right = sum(per_fz[g] for g in FOOT_CONTACT_GEOMS if g.startswith("right"))
    return left, right, per_fz, per_fx


def _contact_vertical_forces_N(
    model: Any, data: Any, ids: dict[str, int]
) -> tuple[float, float]:
    """Backward-compatible per-side vertical contact force accessor."""

    left, right, _, _ = _contact_forces(model, data, ids)
    return left, right


def _force_plate_measurement(
    left_raw_N: float, right_raw_N: float, params: dict[str, Any]
) -> tuple[float, float]:
    raw_total = max(0.0, left_raw_N + right_raw_N)
    measured_total = max(
        0.0,
        raw_total * float(params["force_plate_scale"])
        + float(params["force_plate_bias_N"]),
    )
    if raw_total <= 1e-12:
        return 0.0, 0.0
    return (
        measured_total * max(0.0, left_raw_N) / raw_total,
        measured_total * max(0.0, right_raw_N) / raw_total,
    )


def _debounced_force_plate_contacts(
    time_s: list[float],
    fz_left_N: list[float],
    fz_right_N: list[float],
    threshold_N: float,
) -> tuple[list[bool], list[bool]]:
    total = [left + right for left, right in zip(fz_left_N, fz_right_N)]
    dt = _median_dt(time_s)
    sustain_n = max(1, int(math.ceil(0.025 / dt)))
    no_contact = [force < threshold_N for force in total]
    takeoff_idx = _first_sustained_index(
        no_contact,
        time_s,
        start_time=PHASES["propulsion"][0],
        sustain_n=sustain_n,
    )
    if takeoff_idx is None:
        return (
            [left > 0.5 * threshold_N for left in fz_left_N],
            [right > 0.5 * threshold_N for right in fz_right_N],
        )
    recontact = [force >= threshold_N for force in total]
    landing_idx = _first_sustained_index(
        recontact,
        time_s,
        start_time=time_s[takeoff_idx] + 0.025,
        sustain_n=sustain_n,
    )
    if landing_idx is None:
        landing_idx = len(time_s)
    contact_state = [
        i < takeoff_idx or i >= landing_idx
        for i in range(len(time_s))
    ]
    return list(contact_state), list(contact_state)


def _bar_measurement_position(
    model: Any, data: Any, ids: dict[str, int], params: dict[str, Any]
) -> tuple[float, float, float]:
    site = [float(x) for x in data.site_xpos[ids["bar_lpt_site"]]]
    attach_x, attach_z = [float(x) for x in params["bar_attachment_offset_m"]]
    offset = [attach_x, 0.0, attach_z]
    if np is not None:
        xmat = np.asarray(data.xmat[ids["bar_body"]], dtype=float).reshape(3, 3)
        adjusted = np.asarray(site, dtype=float) + xmat @ np.asarray(offset, dtype=float)
        return tuple(float(x) for x in adjusted)
    return site[0] + offset[0], site[1] + offset[1], site[2] + offset[2]


def _qpos_value(model: Any, data: Any, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"missing MuJoCo joint: {joint_name}")
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _qvel_value(model: Any, data: Any, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"missing MuJoCo joint: {joint_name}")
    return float(data.qvel[model.jnt_dofadr[joint_id]])


def _vector_norm(values: Any) -> float:
    if np is not None:
        return float(np.linalg.norm(values))
    return math.sqrt(sum(float(x) * float(x) for x in values))


def _assert_finite_mujoco_state(data: Any) -> None:
    for label, values in (("qpos", data.qpos), ("qvel", data.qvel), ("ctrl", data.ctrl)):
        for value in values:
            if not math.isfinite(float(value)):
                raise ValueError(f"nonfinite MuJoCo {label}")


def _validate_rollout_result(
    traces: dict[str, Any],
    events: dict[str, Any],
    summary: dict[str, float],
    diagnostics: dict[str, Any],
) -> None:
    if diagnostics.get("used_mujoco") is not True:
        raise ValueError("rollout did not use MuJoCo-primary simulation")
    for key in CANONICAL_TRACE_KEYS:
        if key not in traces:
            raise ValueError(f"missing trace key after MuJoCo rollout: {key}")
        if key.endswith("_contact"):
            continue
        for value in _float_list(traces[key]):
            if not math.isfinite(value):
                raise ValueError(f"nonfinite trace value after MuJoCo rollout: {key}")
    if not bool(events.get("sustained_no_foot_contact")):
        raise ValueError("MuJoCo rollout did not produce sustained no-foot-contact")
    if events.get("landing_index") is None:
        raise ValueError("MuJoCo rollout did not produce landing recontact")
    if not bool(events.get("phase_order_valid")):
        raise ValueError("MuJoCo event order is invalid")
    if float(events.get("airborne_duration_s", 0.0)) < 0.040:
        raise ValueError("MuJoCo airborne duration is too short")
    if float(diagnostics.get("qvel_norm_max", math.inf)) > MAX_QVEL_NORM:
        raise ValueError("MuJoCo qvel norm exceeded stability gate")
    if float(diagnostics.get("root_x_abs_max_m", math.inf)) > MAX_ROOT_ABS_X_M:
        raise ValueError("MuJoCo root x exceeded stability gate")
    if float(diagnostics.get("root_z_min_m", -math.inf)) < MIN_ROOT_Z_M:
        raise ValueError("MuJoCo root z below stability gate")
    if float(diagnostics.get("root_z_max_m", math.inf)) > MAX_ROOT_Z_M:
        raise ValueError("MuJoCo root z above stability gate")
    if float(diagnostics.get("root_pitch_abs_max_rad", math.inf)) > MAX_ROOT_ABS_PITCH_RAD:
        raise ValueError("MuJoCo root pitch exceeded stability gate")
    if float(diagnostics.get("fz_total_peak_N", math.inf)) > MAX_FZ_TOTAL_N:
        raise ValueError("MuJoCo force plate peak exceeded stability gate")
    net_impulse = _integrate(
        _float_list(traces["time_s"]),
        _float_list(traces["fnet_N"]),
        0,
        len(_float_list(traces["time_s"])) - 1,
    )
    if abs(net_impulse) > MAX_ABS_NET_IMPULSE_NS:
        raise ValueError("MuJoCo net force impulse exceeded stability gate")
    if summary["lpt_tether_impulse_fraction_of_propulsive_impulse"] > MAX_LPT_IMPULSE_FRACTION:
        raise ValueError("LPT tether impulse fraction exceeded diagnostic gate")


def _phase_x(t: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0
    return _clamp((t - start) / (end - start), 0.0, 1.0)


def _smoothstep(x: float) -> float:
    x = _clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft_schema_map(value: float, schema_lo: float, schema_hi: float, safe_lo: float, safe_hi: float) -> float:
    """Map a valid schema coordinate monotonically into a stable PD band."""

    if not all(math.isfinite(x) for x in (value, schema_lo, schema_hi, safe_lo, safe_hi)):
        raise ValueError("non-finite soft controller mapping input")
    if not schema_lo < schema_hi or not safe_lo < safe_hi:
        raise ValueError("invalid soft controller mapping interval")
    normalized = 2.0 * (value - schema_lo) / (schema_hi - schema_lo) - 1.0
    # tanh keeps the controller bounded while retaining a nonzero derivative
    # throughout the schema domain, including the intended evidence region.
    shape = 1.35
    mapped = math.tanh(shape * normalized) / math.tanh(shape)
    return 0.5 * (safe_lo + safe_hi) + 0.5 * (safe_hi - safe_lo) * mapped


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _sequence(values: Any) -> list[Any]:
    if np is not None:
        try:
            if isinstance(values, np.ndarray):
                return values.tolist()
        except Exception:
            pass
    if isinstance(values, tuple):
        return list(values)
    return values


def _float_list(values: Any) -> list[float]:
    return [float(x) for x in _sequence(values)]


def _bool_list(values: Any) -> list[bool]:
    return [bool(x) for x in _sequence(values)]


def _maybe_array(values: Any) -> Any:
    if np is None:
        return list(values) if isinstance(values, tuple) else values
    if isinstance(values, list) and values and isinstance(values[0], bool):
        return np.asarray(values, dtype=bool)
    if isinstance(values, list):
        return np.asarray(values, dtype=float)
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def _median_dt(time_s: list[float]) -> float:
    if len(time_s) < 2:
        return DT
    diffs = [b - a for a, b in zip(time_s[:-1], time_s[1:]) if b > a]
    if not diffs:
        return DT
    diffs.sort()
    return diffs[len(diffs) // 2]


def _first_index_after(time_s: list[float], t0: float) -> int:
    for i, t in enumerate(time_s):
        if t >= t0:
            return i
    return len(time_s) - 1


def _first_sustained_index(
    flags: list[bool],
    time_s: list[float],
    start_time: float,
    sustain_n: int,
) -> int | None:
    start_idx = _first_index_after(time_s, start_time)
    for i in range(start_idx, max(start_idx, len(flags) - sustain_n + 1)):
        if all(flags[i : i + sustain_n]):
            return i
    return None


def _integrate(time_s: list[float], values: list[float], i0: int, i1: int) -> float:
    if i1 <= i0:
        return 0.0
    i0 = max(0, i0)
    i1 = min(i1, len(values) - 1)
    total = 0.0
    for i in range(i0 + 1, i1 + 1):
        dt = time_s[i] - time_s[i - 1]
        total += 0.5 * (values[i] + values[i - 1]) * dt
    return total


def _differentiate(time_s: list[float], values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    out = [0.0 for _ in values]
    for i in range(1, len(values) - 1):
        dt = time_s[i + 1] - time_s[i - 1]
        out[i] = (values[i + 1] - values[i - 1]) / dt if dt > 0 else 0.0
    first_dt = time_s[1] - time_s[0]
    last_dt = time_s[-1] - time_s[-2]
    out[0] = (values[1] - values[0]) / first_dt if first_dt > 0 else 0.0
    out[-1] = (values[-1] - values[-2]) / last_dt if last_dt > 0 else 0.0
    return out


def _first_order_filter(values: list[float], time_s: list[float], tau: float) -> list[float]:
    if tau <= 0.0 or len(values) < 2:
        return list(values)
    out = [values[0]]
    for i in range(1, len(values)):
        dt = max(time_s[i] - time_s[i - 1], 0.0)
        alpha = dt / (tau + dt)
        out.append(out[-1] + alpha * (values[i] - out[-1]))
    return out


def _delay(values: list[float], steps: int) -> list[float]:
    if steps <= 0:
        return list(values)
    if not values:
        return []
    return [values[0]] * steps + values[:-steps]
