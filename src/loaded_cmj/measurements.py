"""Source-authoritative measurement channels and summary extraction."""

from __future__ import annotations

from .plant import (
    CANONICAL_SUMMARY_KEYS,
    CANONICAL_TRACE_KEYS,
    EXTENDED_KINEMATICS_TRACE_KEYS,
    NEW_TELEMETRY_TRACE_KEYS,
    extract_traces,
    summarize_trial,
)


MEASUREMENT_CHANNELS = {
    "force_platform": {
        "channels": ("fz_left_N", "fz_right_N", "fz_total_N", "fnet_N"),
        "units": "N",
        "description": "Bilateral vertical force-platform channels and their net total.",
    },
    "bar_lpt": {
        "channels": ("bar_displacement_m", "bar_velocity_m_s", "lpt_tether_force_N"),
        "units": "m, m/s, N",
        "description": "Linear-position transducer measurement of the loaded bar itself.",
    },
    "state": {
        "channels": ("root_z_m", "root_pitch_rad", "root_pitch_rate_rad_s"),
        "units": "m, rad, rad/s",
        "description": "Model state channels used for event and mechanics diagnostics.",
    },
    "kinematics": {
        "channels": EXTENDED_KINEMATICS_TRACE_KEYS,
        "units": "native SI/radian state units",
        "description": "Read-only COM, joint, and bar-rack qpos/qvel traces from the same MuJoCo rollout.",
    },
}


__all__ = [
    "CANONICAL_SUMMARY_KEYS",
    "CANONICAL_TRACE_KEYS",
    "EXTENDED_KINEMATICS_TRACE_KEYS",
    "NEW_TELEMETRY_TRACE_KEYS",
    "MEASUREMENT_CHANNELS",
    "extract_traces",
    "summarize_trial",
]
