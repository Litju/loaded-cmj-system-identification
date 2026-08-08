"""Render the publication static figures from one authoritative rollout.

This module is presentation-only.  It reads rollout traces, scalar summaries,
events, phases, and public observations without changing, resampling, or
recomputing scientific measurements.  The palette encodes physical channels;
line style encodes side where appropriate; sparse markers encode observations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from loaded_cmj.dataset import load_trial
from loaded_cmj.parameters import load_named_parameters
from loaded_cmj.plant import PHASE_NAMES
from loaded_cmj.simulation import simulate_trial


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
OUTPUT_DIR = MEDIA
SCENARIO_ID = "20kg_nominal_a"

# One publication palette.  Color identifies the physical channel or region;
# source and side are carried by independent geometric channels.
BACKGROUND = "#0B1118"
PANEL = "#101720"
TEXT = "#F2F4F8"
MUTED = "#B7BDC8"
GRID = "#34404C"
SPINE = "#6B7785"
EVENT = "#F2F4F8"

BLUE = "#0072B2"                 # hip / generic model channel
SKY_BLUE = "#56B4E9"             # left side / trunk / left plate
GREEN = "#009E73"                 # ankle / forefoot / velocity
ORANGE = "#E69F00"               # right side / MTP / toe
VERMILLION = "#D55E00"           # knee
PURPLE = "#CC79A7"               # rocker / displacement
WARM_GOLD = "#F0C75E"             # COM
OFF_WHITE = "#F2F4F8"             # aggregate / total
NET = "#9B8AFB"                  # net GRF
AUX_SHOULDER = "#B7BDC8"
AUX_ELBOW = "#7E8A97"

COLORS = {
    "hip": BLUE,
    "knee": VERMILLION,
    "ankle": GREEN,
    "rocker": PURPLE,
    "mtp": ORANGE,
    "lumbar": SKY_BLUE,
    "shoulder": AUX_SHOULDER,
    "elbow": AUX_ELBOW,
    "heel": BLUE,
    "forefoot": GREEN,
    "toe": ORANGE,
    "left": SKY_BLUE,
    "right": ORANGE,
    "total": OFF_WHITE,
    "net": NET,
    "com": WARM_GOLD,
    "lpt_displacement": PURPLE,
    "lpt_velocity": GREEN,
    "tether": MUTED,
    "root": SKY_BLUE,
    "bar": ORANGE,
}

PHASE_COLORS = (
    "#28506D",  # weighing
    "#5E4268",  # unweighting
    "#7E4D36",  # braking
    "#876D24",  # propulsion
    "#244B6B",  # flight
    "#553675",  # landing absorption
    "#245E58",  # stabilization
)
EVENT_COLORS = {
    "movement_onset_time_s": WARM_GOLD,
    "takeoff_time_s": SKY_BLUE,
    "landing_time_s": VERMILLION,
}
EVENT_LABELS = {
    "movement_onset_time_s": "onset",
    "takeoff_time_s": "takeoff",
    "landing_time_s": "landing",
}

COMMON_PLOT_FAMILY = (
    "observable_fit.png",
    "combined_grf_com_lpt.png",
    "force_plate_metrics.png",
    "global_kinematics.png",
    "foot_kinematics.png",
    "joint_kinematics.png",
    "auxiliary_kinematics.png",
    "contact_mechanics.png",
    "phase_events.png",
    "summary_metrics.png",
)


def _finite(values: Iterable[Any]) -> np.ndarray:
    """Return finite numeric values, preserving source values otherwise."""

    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return np.asarray(out, dtype=float)


def _limits(values: Iterable[Any], *, padding: float = 0.06) -> tuple[float, float] | None:
    data = _finite(values)
    if data.size == 0:
        return None
    low = float(np.min(data))
    high = float(np.max(data))
    span = high - low
    pad = max(span * padding, abs(low) * 0.02, abs(high) * 0.02, 1e-6)
    if span == 0.0:
        pad = max(abs(low) * 0.08, 0.02)
    return low - pad, high + pad


def _add_scale(scales: dict[str, list[float]], group: str, values: Iterable[Any]) -> None:
    values_array = _finite(values)
    if values_array.size:
        scales[group].extend(values_array.tolist())


def build_scale_context(scenarios: Iterable[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Build complete-release display limits from all supplied frozen scenarios."""

    values: dict[str, list[float]] = defaultdict(list)
    for entry in scenarios:
        result = entry.get("result", entry)
        trial = entry.get("trial", {})
        traces = result.get("traces", {})
        observed = trial.get("observations", {})
        for key in ("fz_left_N", "fz_right_N", "fz_total_N", "total_fz_N", "fnet_N"):
            _add_scale(values, "force_N", traces.get(key, []))
        for key in ("fz_left_N", "fz_right_N", "fz_total_N"):
            _add_scale(values, "force_N", observed.get(key, []))
        _add_scale(values, "lpt_displacement_m", traces.get("bar_displacement_m", []))
        _add_scale(values, "lpt_displacement_m", observed.get("bar_displacement_m", []))
        _add_scale(values, "lpt_velocity_m_s", traces.get("bar_velocity_m_s", []))
        _add_scale(values, "lpt_velocity_m_s", observed.get("bar_velocity_m_s", []))
        _add_scale(values, "com_z_m", traces.get("com_z_m", []))
        for key in ("root_x_m", "com_x_m"):
            _add_scale(values, "horizontal_position_m", traces.get(key, []))
        for key in ("root_z_m", "com_z_m"):
            _add_scale(values, "vertical_position_m", traces.get(key, []))
        for key in ("bar_z_m", "bar_displacement_m"):
            _add_scale(values, "bar_position_m", traces.get(key, []))
        for key in ("root_x_velocity_m_s", "root_z_velocity_m_s", "bar_velocity_m_s"):
            _add_scale(values, "velocity_m_s", traces.get(key, []))
        for key in (
            "left_heel_z_m", "left_forefoot_z_m", "left_toe_z_m",
            "right_heel_z_m", "right_forefoot_z_m", "right_toe_z_m",
        ):
            _add_scale(values, "foot_height_m", traces.get(key, []))
        _add_scale(values, "foot_clearance_m", traces.get("foot_clearance_m", []))
        for joint in ("hip", "knee", "ankle", "rocker", "mtp"):
            position_keys = (
                (f"left_{joint}_rad", f"right_{joint}_rad")
                if joint in {"hip", "knee", "ankle"}
                else (f"left_forefoot_rocker_rad", f"right_forefoot_rocker_rad")
                if joint == "rocker"
                else ("left_mtp_rad", "right_mtp_rad")
            )
            velocity_keys = (
                (f"left_{joint}_velocity_rad_s", f"right_{joint}_velocity_rad_s")
                if joint in {"hip", "knee", "ankle"}
                else ("left_forefoot_rocker_velocity_rad_s", "right_forefoot_rocker_velocity_rad_s")
                if joint == "rocker"
                else ("left_mtp_velocity_rad_s", "right_mtp_velocity_rad_s")
            )
            for key in position_keys:
                _add_scale(values, f"{joint}_position_rad", traces.get(key, []))
            for key in velocity_keys:
                _add_scale(values, f"{joint}_velocity_rad_s", traces.get(key, []))
        for key in (
            "left_heel_fz_N", "left_forefoot_fz_N", "left_toe_fz_N",
            "right_heel_fz_N", "right_forefoot_fz_N", "right_toe_fz_N",
        ):
            _add_scale(values, "regional_fz_N", traces.get(key, []))
        for key in (
            "left_heel_fx_N", "left_forefoot_fx_N", "left_toe_fx_N",
            "right_heel_fx_N", "right_forefoot_fx_N", "right_toe_fx_N",
        ):
            _add_scale(values, "regional_fx_N", traces.get(key, []))
        _add_scale(values, "cop_x_m", traces.get("cop_x_m", []))
        for key in ("left_slip_vx_m_s", "right_slip_vx_m_s"):
            _add_scale(values, "slip_m_s", traces.get(key, []))
        for key in ("lpt_tether_force_N",):
            _add_scale(values, "tether_N", traces.get(key, []))
        for key in ("bar_rack_x_m", "bar_rack_z_m"):
            _add_scale(values, "bar_rack_translation_m", traces.get(key, []))
        for key in ("bar_rack_x_velocity_m_s", "bar_rack_z_velocity_m_s"):
            _add_scale(values, "bar_rack_translation_velocity_m_s", traces.get(key, []))
        for key in ("lumbar_pitch_rad", "bar_rack_pitch_rad"):
            _add_scale(values, "pitch_rad", traces.get(key, []))
        for key in ("lumbar_pitch_rate_rad_s", "bar_rack_pitch_velocity_rad_s"):
            _add_scale(values, "pitch_rate_rad_s", traces.get(key, []))
        for key in ("left_shoulder_rad", "right_shoulder_rad", "left_elbow_rad", "right_elbow_rad"):
            _add_scale(values, "upper_angles_rad", traces.get(key, []))
        for key in (
            "left_shoulder_velocity_rad_s", "right_shoulder_velocity_rad_s",
            "left_elbow_velocity_rad_s", "right_elbow_velocity_rad_s",
        ):
            _add_scale(values, "upper_rates_rad_s", traces.get(key, []))
        _add_scale(values, "time_s", traces.get("time_s", []))

    context: dict[str, tuple[float, float]] = {}
    for name, data in values.items():
        limit = _limits(data)
        if limit is not None:
            context[name] = limit
    if "time_s" in context:
        data = values["time_s"]
        context["time_s"] = (float(min(data)), float(max(data)))
    return context


def _limit(scales: dict[str, tuple[float, float]], group: str, values: Iterable[Any]) -> tuple[float, float] | None:
    return scales.get(group) or _limits(values)


def _configure_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BACKGROUND,
        "savefig.edgecolor": BACKGROUND,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": SPINE,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
    })


def _style_axis(ax: Any, title: str, ylabel: str | None = None, *, limits: tuple[float, float] | None = None) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, loc="left", color=TEXT, pad=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT)
    if limits is not None:
        ax.set_ylim(*limits)
    ax.tick_params(axis="both", colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.7)
    ax.grid(True, color=GRID, alpha=0.48, linewidth=0.55)
    ax.set_axisbelow(True)


def _style_legend(legend: Any) -> None:
    legend.get_frame().set_facecolor(BACKGROUND)
    legend.get_frame().set_edgecolor(GRID)
    legend.get_frame().set_alpha(0.94)
    if legend.get_title() is not None:
        legend.get_title().set_color(MUTED)
    for text in legend.get_texts():
        text.set_color(TEXT)


def _add_legend(ax: Any, handles: list[Any], title: str, *, loc: str = "upper left", ncol: int = 1) -> Any:
    previous = ax.get_legend()
    if previous is not None:
        ax.add_artist(previous)
    legend = ax.legend(handles=handles, title=title, loc=loc, ncol=ncol, borderpad=0.42, handlelength=2.0)
    _style_legend(legend)
    return legend


def _source_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=TEXT, linewidth=2.0, label="MuJoCo / model"),
        Line2D([], [], color=TEXT, linewidth=0.8, marker="o", markersize=4.0,
               markerfacecolor=BACKGROUND, markeredgewidth=0.9, label="Observed"),
    ]


def _side_handles(*, colors: bool = False) -> list[Line2D]:
    return [
        Line2D([], [], color=SKY_BLUE if colors else TEXT, linewidth=2.0, linestyle="-", label="Left"),
        Line2D([], [], color=ORANGE if colors else TEXT, linewidth=2.0, linestyle="--", label="Right"),
    ]


def _region_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=COLORS["heel"], linewidth=2.0, label="Heel"),
        Line2D([], [], color=COLORS["forefoot"], linewidth=2.0, label="Forefoot"),
        Line2D([], [], color=COLORS["toe"], linewidth=2.0, label="Toe / MTP"),
    ]


def _joint_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=COLORS["hip"], linewidth=2.0, label="Hip"),
        Line2D([], [], color=COLORS["knee"], linewidth=2.0, label="Knee"),
        Line2D([], [], color=COLORS["ankle"], linewidth=2.0, label="Ankle"),
        Line2D([], [], color=COLORS["rocker"], linewidth=2.0, label="Rocker"),
        Line2D([], [], color=COLORS["mtp"], linewidth=2.0, label="MTP"),
    ]


def _time_values(result: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    traces = result["traces"]
    return np.asarray(traces["time_s"], dtype=float), traces


def _plot_model(
    ax: Any,
    time_s: Iterable[float],
    values: Iterable[float],
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.85,
    alpha: float = 0.95,
) -> Any:
    return ax.plot(
        np.asarray(list(time_s), dtype=float),
        np.asarray(list(values), dtype=float),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=3,
    )[0]


def _plot_observed(
    ax: Any,
    time_s: Iterable[float],
    values: Iterable[float],
    *,
    color: str,
    linewidth: float = 0.72,
    alpha: float = 0.92,
) -> Any:
    time = np.asarray(list(time_s), dtype=float)
    series = np.asarray(list(values), dtype=float)
    markevery = max(1, len(series) // 18)
    return ax.plot(
        time,
        series,
        color=color,
        linestyle="-",
        linewidth=linewidth,
        marker="o",
        markersize=3.5,
        markevery=markevery,
        markerfacecolor=BACKGROUND,
        markeredgecolor=color,
        markeredgewidth=0.9,
        alpha=alpha,
        zorder=5,
    )[0]


def _phase_timing(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("phase_timing_s", {})


def _phase_context(ax: Any, result: dict[str, Any], *, labels: bool = False, strong: bool = False) -> None:
    """Show quiet phase ribbons and clear event lines without filling the plot."""

    timing = _phase_timing(result)
    for index, phase_name in enumerate(PHASE_NAMES):
        start_end = timing.get(phase_name.lower())
        if not start_end:
            continue
        start, end = float(start_end[0]), float(start_end[1])
        if strong:
            ax.axvspan(start, end, color=PHASE_COLORS[index], alpha=0.16, linewidth=0, zorder=0)
        else:
            ax.fill_between(
                [start, end], [0.94, 0.94], [0.995, 0.995],
                transform=ax.get_xaxis_transform(), color=PHASE_COLORS[index],
                alpha=0.92, linewidth=0, zorder=0,
            )
        if labels:
            ax.text(
                (start + end) / 2.0, 0.968, phase_name.replace("_", " "),
                transform=ax.get_xaxis_transform(), color=MUTED,
                fontsize=7.2, ha="center", va="center", rotation=90 if "_" in phase_name else 0,
                clip_on=True, zorder=1,
            )

    events = result.get("events", {})
    for key, label in EVENT_LABELS.items():
        value = events.get(key)
        if value is None:
            continue
        time_s = float(value)
        color = EVENT_COLORS[key]
        ax.axvline(time_s, color=color, linewidth=0.85, linestyle=(0, (3, 2)), alpha=0.92, zorder=2)
        if labels:
            ax.text(
                time_s, 0.04, label, transform=ax.get_xaxis_transform(), color=color,
                fontsize=7.2, ha="left", va="bottom", rotation=90, clip_on=True, zorder=4,
            )


def _set_time_axis(axes: Iterable[Any], scales: dict[str, tuple[float, float]]) -> None:
    limits = scales.get("time_s")
    if limits is not None:
        for ax in axes:
            ax.set_xlim(*limits)


def _finish(fig: Any, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.text(
        0.995, 0.006, f"{SCENARIO_ID} | fixed 20 kg | production rollout",
        ha="right", va="bottom", color=MUTED, fontsize=7,
    )
    fig.savefig(OUTPUT_DIR / filename, dpi=170, facecolor=BACKGROUND, edgecolor=BACKGROUND)
    plt.close(fig)


def _observable_fit(result: dict[str, Any], observed: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    observed_time = np.asarray(observed["time_s"], dtype=float)
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ observable fit — model and observations", color=TEXT, fontsize=14, fontweight="bold")

    _phase_context(axes[0], result)
    _plot_model(axes[0], time_s, traces["fz_left_N"], color=COLORS["left"])
    _plot_model(axes[0], time_s, traces["fz_right_N"], color=COLORS["right"], linestyle="--")
    _plot_observed(axes[0], observed_time, observed["fz_left_N"], color=COLORS["left"])
    _plot_observed(axes[0], observed_time, observed["fz_right_N"], color=COLORS["right"])
    _style_axis(axes[0], "Bilateral force plates", "force (N)", limits=_limit(scales, "force_N", traces["fz_left_N"]))
    _add_legend(axes[0], _side_handles(colors=True), "SIDE")
    _add_legend(axes[0], _source_handles(), "SOURCE", loc="upper right")

    _phase_context(axes[1], result)
    _plot_model(axes[1], time_s, traces["fz_total_N"], color=COLORS["total"], linewidth=2.25)
    _plot_observed(axes[1], observed_time, observed["fz_total_N"], color=COLORS["total"])
    _plot_model(axes[1], time_s, traces["fnet_N"], color=COLORS["net"], linestyle="--")
    _style_axis(axes[1], "Total and net ground-reaction force", "force (N)", limits=_limit(scales, "force_N", traces["fz_total_N"]))
    _add_legend(axes[1], _source_handles(), "SOURCE")
    _add_legend(axes[1], [Line2D([], [], color=OFF_WHITE, linewidth=2.2, label="Total Fz"),
                          Line2D([], [], color=NET, linewidth=1.8, linestyle="--", label="Net GRF")],
                "CHANNEL", loc="upper right")

    _phase_context(axes[2], result)
    _plot_model(axes[2], time_s, traces["bar_displacement_m"], color=COLORS["lpt_displacement"])
    _plot_observed(axes[2], observed_time, observed["bar_displacement_m"], color=COLORS["lpt_displacement"])
    _style_axis(axes[2], "Bar/LPT displacement — bar displacement, not COM displacement", "displacement (m)",
                limits=_limit(scales, "lpt_displacement_m", traces["bar_displacement_m"]))
    _add_legend(axes[2], _source_handles(), "SOURCE", loc="upper right")

    _phase_context(axes[3], result)
    _plot_model(axes[3], time_s, traces["bar_velocity_m_s"], color=COLORS["lpt_velocity"])
    _plot_observed(axes[3], observed_time, observed["bar_velocity_m_s"], color=COLORS["lpt_velocity"])
    _style_axis(axes[3], "Bar/LPT velocity", "velocity (m/s)", limits=_limit(scales, "lpt_velocity_m_s", traces["bar_velocity_m_s"]))
    _add_legend(axes[3], _source_handles(), "SOURCE", loc="lower right")

    _phase_context(axes[4], result, labels=True)
    _plot_model(axes[4], time_s, traces["lpt_tether_force_N"], color=COLORS["tether"])
    _style_axis(axes[4], "LPT tether diagnostic", "force (N)", limits=_limit(scales, "tether_N", traces["lpt_tether_force_N"]))
    _add_legend(axes[4], [Line2D([], [], color=COLORS["tether"], linewidth=2, label="MuJoCo / model")], "SOURCE", loc="lower right")
    axes[4].set_xlabel("time (s)", color=TEXT)
    _set_time_axis(axes, scales)
    _finish(fig, "observable_fit.png")


def _combined_grf_com_lpt(result: dict[str, Any], observed: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    observed_time = np.asarray(observed["time_s"], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10.5), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ — synchronized GRF, COM, and LPT channels", color=TEXT, fontsize=14, fontweight="bold")

    _phase_context(axes[0], result)
    _plot_model(axes[0], time_s, traces["fz_total_N"], color=COLORS["total"], linewidth=2.45)
    _plot_observed(axes[0], observed_time, observed["fz_total_N"], color=COLORS["total"])
    _plot_model(axes[0], time_s, traces["fnet_N"], color=COLORS["net"], linestyle="--")
    _style_axis(axes[0], "GRF", "force (N)", limits=_limit(scales, "force_N", traces["fz_total_N"]))
    _add_legend(axes[0], _source_handles(), "SOURCE")

    _phase_context(axes[1], result)
    _plot_model(axes[1], time_s, traces["com_z_m"], color=COLORS["com"], linewidth=2.0)
    _style_axis(axes[1], "Centre of mass vertical position", "position (m)", limits=_limit(scales, "com_z_m", traces["com_z_m"]))
    _add_legend(axes[1], [Line2D([], [], color=COLORS["com"], linewidth=2, label="MuJoCo / model")], "SOURCE")

    _phase_context(axes[2], result)
    _plot_model(axes[2], time_s, traces["bar_displacement_m"], color=COLORS["lpt_displacement"], linewidth=2.0)
    _plot_observed(axes[2], observed_time, observed["bar_displacement_m"], color=COLORS["lpt_displacement"])
    _style_axis(axes[2], "Bar/LPT displacement — bar displacement, not COM displacement", "displacement (m)",
                limits=_limit(scales, "lpt_displacement_m", traces["bar_displacement_m"]))
    _add_legend(axes[2], _source_handles(), "SOURCE")

    _phase_context(axes[3], result, labels=True)
    _plot_model(axes[3], time_s, traces["bar_velocity_m_s"], color=COLORS["lpt_velocity"], linewidth=2.0)
    _plot_observed(axes[3], observed_time, observed["bar_velocity_m_s"], color=COLORS["lpt_velocity"])
    _style_axis(axes[3], "Bar/LPT velocity", "velocity (m/s)", limits=_limit(scales, "lpt_velocity_m_s", traces["bar_velocity_m_s"]))
    _add_legend(axes[3], _source_handles(), "SOURCE", loc="lower right")
    axes[3].set_xlabel("time (s)", color=TEXT)
    _set_time_axis(axes, scales)
    _finish(fig, "combined_grf_com_lpt.png")


def _force_plate_metrics(result: dict[str, Any], observed: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    observed_time = np.asarray(observed["time_s"], dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11.5), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ force-plate mechanics", color=TEXT, fontsize=14, fontweight="bold")

    _phase_context(axes[0], result)
    for key, color, style in (("fz_left_N", COLORS["left"], "-"), ("fz_right_N", COLORS["right"], "--"), ("fz_total_N", COLORS["total"], "-")):
        _plot_model(axes[0], time_s, traces[key], color=color, linestyle=style, linewidth=2.35 if key == "fz_total_N" else 1.8)
        _plot_observed(axes[0], observed_time, observed[key], color=color)
    _style_axis(axes[0], "Bilateral vertical force plates", "force (N)", limits=_limit(scales, "force_N", traces["fz_total_N"]))
    _add_legend(axes[0], _side_handles(colors=True), "SIDE")
    _add_legend(axes[0], _source_handles(), "SOURCE", loc="upper right")

    _phase_context(axes[1], result)
    regional = (
        ("left_heel_fz_N", COLORS["heel"], "-"), ("right_heel_fz_N", COLORS["heel"], "--"),
        ("left_forefoot_fz_N", COLORS["forefoot"], "-"), ("right_forefoot_fz_N", COLORS["forefoot"], "--"),
        ("left_toe_fz_N", COLORS["toe"], "-"), ("right_toe_fz_N", COLORS["toe"], "--"),
    )
    for key, color, style in regional:
        _plot_model(axes[1], time_s, traces[key], color=color, linestyle=style, linewidth=1.4, alpha=0.88)
    _style_axis(axes[1], "Regional vertical contact force", "force (N)", limits=_limit(scales, "regional_fz_N", traces["left_heel_fz_N"]))
    _add_legend(axes[1], _region_handles(), "REGION")
    _add_legend(axes[1], _side_handles(), "SIDE", loc="upper right")

    _phase_context(axes[2], result)
    regional_horizontal = (
        ("left_heel_fx_N", COLORS["heel"], "-"), ("right_heel_fx_N", COLORS["heel"], "--"),
        ("left_forefoot_fx_N", COLORS["forefoot"], "-"), ("right_forefoot_fx_N", COLORS["forefoot"], "--"),
        ("left_toe_fx_N", COLORS["toe"], "-"), ("right_toe_fx_N", COLORS["toe"], "--"),
    )
    for key, color, style in regional_horizontal:
        _plot_model(axes[2], time_s, traces[key], color=color, linestyle=style, linewidth=1.4, alpha=0.88)
    _style_axis(axes[2], "Regional horizontal contact force", "force (N)", limits=_limit(scales, "regional_fx_N", traces["left_heel_fx_N"]))
    _add_legend(axes[2], _region_handles(), "REGION")
    _add_legend(axes[2], _side_handles(), "SIDE", loc="upper right")

    _phase_context(axes[3], result, labels=True)
    _plot_model(axes[3], time_s, traces["total_fx_N"], color=COLORS["net"], linewidth=2.2)
    _style_axis(axes[3], "Total horizontal force-plate signal", "force (N)", limits=_limit(scales, "regional_fx_N", traces["total_fx_N"]))
    _add_legend(axes[3], [Line2D([], [], color=COLORS["net"], linewidth=2, label="Total Fx")], "CHANNEL", loc="lower right")
    axes[3].set_xlabel("time (s)", color=TEXT)
    _set_time_axis(axes, scales)
    _finish(fig, "force_plate_metrics.png")


def _global_kinematics(result: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11.5), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ global kinematics", color=TEXT, fontsize=14, fontweight="bold")
    panels = (
        ((("root_x_m", COLORS["root"], "-"), ("com_x_m", COLORS["com"], "-")), "Horizontal position", "position (m)"),
        ((("root_z_m", COLORS["root"], "-"), ("com_z_m", COLORS["com"], "-")), "Vertical position", "position (m)"),
        ((("bar_z_m", COLORS["bar"], "-"), ("bar_displacement_m", COLORS["lpt_displacement"], "--")), "Loaded bar position channels", "position (m)"),
        ((("root_x_velocity_m_s", COLORS["root"], "-"), ("root_z_velocity_m_s", COLORS["root"], "--"), ("bar_velocity_m_s", COLORS["lpt_velocity"], "-")), "Global velocities", "velocity (m/s)"),
    )
    for index, (series, title, ylabel) in enumerate(panels):
        _phase_context(axes[index], result, labels=index == len(panels) - 1)
        for key, color, style in series:
            _plot_model(axes[index], time_s, traces[key], color=color, linestyle=style)
        group = ("horizontal_position_m", "vertical_position_m", "bar_position_m", "velocity_m_s")[index]
        _style_axis(axes[index], title, ylabel, limits=_limit(scales, group, traces[series[0][0]]))
        labels = [
            "Root x", "COM x", "Root z", "COM z", "Bar z", "Bar displacement",
            "Root x velocity", "Root z velocity", "Bar velocity",
        ]
        offset = (0, 2, 4, 6)[index]
        _add_legend(axes[index], [Line2D([], [], color=color, linestyle=style, linewidth=2, label=labels[offset + i])
                                 for i, (_, color, style) in enumerate(series)], "CHANNEL")
    axes[-1].set_xlabel("time (s)", color=TEXT)
    _set_time_axis(axes, scales)
    _finish(fig, "global_kinematics.png")


def _foot_kinematics(result: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ foot kinematics", color=TEXT, fontsize=14, fontweight="bold")
    regions = (("heel", "left_heel_z_m"), ("forefoot", "left_forefoot_z_m"), ("toe", "left_toe_z_m"))
    for index, (side, keys) in enumerate((
        ("Left", tuple(f"left_{region}_z_m" for region in ("heel", "forefoot", "toe"))),
        ("Right", tuple(f"right_{region}_z_m" for region in ("heel", "forefoot", "toe"))),
    )):
        _phase_context(axes[index], result)
        for key, region in zip(keys, ("heel", "forefoot", "toe")):
            _plot_model(axes[index], time_s, traces[key], color=COLORS[region], linestyle="-" if side == "Left" else "--")
        _style_axis(axes[index], f"{side} foot landmark heights", "height (m)", limits=_limit(scales, "foot_height_m", traces[keys[0]]))
        _add_legend(axes[index], _region_handles(), "REGION")
    _phase_context(axes[2], result, labels=True)
    _plot_model(axes[2], time_s, traces["foot_clearance_m"], color=COLORS["total"], linewidth=2.25)
    _style_axis(axes[2], "Minimum bilateral foot clearance", "clearance (m)", limits=_limit(scales, "foot_clearance_m", traces["foot_clearance_m"]))
    _add_legend(axes[2], [Line2D([], [], color=OFF_WHITE, linewidth=2, label="Minimum bilateral clearance")], "CHANNEL", loc="lower right")
    axes[2].set_xlabel("time (s)", color=TEXT)
    _set_time_axis(axes, scales)
    _finish(fig, "foot_kinematics.png")


def _joint_kinematics(result: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    rows = (
        ("Hip", "hip", "left_hip_rad", "right_hip_rad", "left_hip_velocity_rad_s", "right_hip_velocity_rad_s", COLORS["hip"]),
        ("Knee", "knee", "left_knee_rad", "right_knee_rad", "left_knee_velocity_rad_s", "right_knee_velocity_rad_s", COLORS["knee"]),
        ("Ankle", "ankle", "left_ankle_rad", "right_ankle_rad", "left_ankle_velocity_rad_s", "right_ankle_velocity_rad_s", COLORS["ankle"]),
        ("Forefoot rocker", "rocker", "left_forefoot_rocker_rad", "right_forefoot_rocker_rad", "left_forefoot_rocker_velocity_rad_s", "right_forefoot_rocker_velocity_rad_s", COLORS["rocker"]),
        ("MTP / toe", "mtp", "left_mtp_rad", "right_mtp_rad", "left_mtp_velocity_rad_s", "right_mtp_velocity_rad_s", COLORS["mtp"]),
    )
    fig, axes = plt.subplots(5, 2, figsize=(13, 15), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ lower-limb joint kinematics", color=TEXT, fontsize=14, fontweight="bold")
    for row, (label, group, left_q, right_q, left_qd, right_qd, color) in enumerate(rows):
        for column, (left_key, right_key, unit, scale_group, title_suffix) in enumerate((
            (left_q, right_q, "rad", f"{group}_position_rad", "position"),
            (left_qd, right_qd, "rad/s", f"{group}_velocity_rad_s", "velocity"),
        )):
            ax = axes[row, column]
            _phase_context(ax, result, labels=row == len(rows) - 1 and column == 1)
            _plot_model(ax, time_s, traces[left_key], color=color, linestyle="-")
            _plot_model(ax, time_s, traces[right_key], color=color, linestyle="--")
            _style_axis(ax, f"{label} · {title_suffix}", unit, limits=_limit(scales, scale_group, traces[left_key]))
            if row == 0 and column == 0:
                _add_legend(ax, _joint_handles(), "JOINT")
                _add_legend(ax, _side_handles(), "SIDE", loc="upper right")
    axes[-1, 0].set_xlabel("time (s)", color=TEXT)
    axes[-1, 1].set_xlabel("time (s)", color=TEXT)
    _set_time_axis(axes.flat, scales)
    _finish(fig, "joint_kinematics.png")


def _auxiliary_kinematics(result: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ auxiliary kinematics", color=TEXT, fontsize=14, fontweight="bold")
    panels = (
        (0, 0, (("lumbar_pitch_rad", COLORS["lumbar"], "-"),), "Trunk pitch", "rad", "pitch_rad", "pitch"),
        (0, 1, (("left_shoulder_rad", COLORS["shoulder"], "-"), ("right_shoulder_rad", COLORS["shoulder"], "--"),
                 ("left_elbow_rad", COLORS["elbow"], "-"), ("right_elbow_rad", COLORS["elbow"], "--")),
         "Shoulder and elbow angles", "rad", "upper_angles_rad", "upper angles"),
        (1, 0, (("lumbar_pitch_rate_rad_s", COLORS["lumbar"], "-"),), "Trunk pitch rate", "rad/s", "pitch_rate_rad_s", "pitch rate"),
        (1, 1, (("left_shoulder_velocity_rad_s", COLORS["shoulder"], "-"), ("right_shoulder_velocity_rad_s", COLORS["shoulder"], "--"),
                 ("left_elbow_velocity_rad_s", COLORS["elbow"], "-"), ("right_elbow_velocity_rad_s", COLORS["elbow"], "--")),
         "Shoulder and elbow angular velocities", "rad/s", "upper_rates_rad_s", "upper rates"),
        (2, 0, (("bar_rack_x_m", COLORS["root"], "-"), ("bar_rack_z_m", COLORS["bar"], "--")),
         "Bar/rack translations", "m", "bar_rack_translation_m", "translation"),
        (2, 1, (("bar_rack_x_velocity_m_s", COLORS["root"], "-"), ("bar_rack_z_velocity_m_s", COLORS["bar"], "--")),
         "Bar/rack translation velocities", "m/s", "bar_rack_translation_velocity_m_s", "translation velocity"),
        (3, 0, (("bar_rack_pitch_rad", COLORS["rocker"], "-"),), "Bar/rack pitch", "rad", "pitch_rad", "pitch"),
        (3, 1, (("bar_rack_pitch_velocity_rad_s", COLORS["rocker"], "-"),), "Bar/rack pitch rate", "rad/s", "pitch_rate_rad_s", "pitch rate"),
    )
    labels = {
        "lumbar_pitch_rad": "Trunk pitch", "lumbar_pitch_rate_rad_s": "Trunk pitch rate",
        "left_shoulder_rad": "Left shoulder", "right_shoulder_rad": "Right shoulder",
        "left_elbow_rad": "Left elbow", "right_elbow_rad": "Right elbow",
        "left_shoulder_velocity_rad_s": "Left shoulder", "right_shoulder_velocity_rad_s": "Right shoulder",
        "left_elbow_velocity_rad_s": "Left elbow", "right_elbow_velocity_rad_s": "Right elbow",
        "bar_rack_x_m": "Rack x", "bar_rack_z_m": "Rack z",
        "bar_rack_x_velocity_m_s": "Rack x velocity", "bar_rack_z_velocity_m_s": "Rack z velocity",
        "bar_rack_pitch_rad": "Rack pitch", "bar_rack_pitch_velocity_rad_s": "Rack pitch rate",
    }
    for row, column, series, title, unit, group, _ in panels:
        ax = axes[row, column]
        _phase_context(ax, result, labels=row == 3 and column == 1)
        for key, color, style in series:
            _plot_model(ax, time_s, traces[key], color=color, linestyle=style)
        _style_axis(ax, title, unit, limits=_limit(scales, group, traces[series[0][0]]))
        _add_legend(ax, [Line2D([], [], color=color, linestyle=style, linewidth=2, label=labels[key])
                         for key, color, style in series], "CHANNEL", ncol=2 if len(series) > 2 else 1)
    _add_legend(axes[0, 0], _side_handles(), "SIDE", loc="upper right")
    axes[-1, 0].set_xlabel("time (s)", color=TEXT)
    axes[-1, 1].set_xlabel("time (s)", color=TEXT)
    _set_time_axis(axes.flat, scales)
    _finish(fig, "auxiliary_kinematics.png")


def _contact_raster(ax: Any, time_s: np.ndarray, traces: dict[str, Any]) -> None:
    rows = (
        ("Left heel", "left_heel_contact", COLORS["heel"]),
        ("Left forefoot", "left_forefoot_contact", COLORS["forefoot"]),
        ("Left toe", "left_toe_contact", COLORS["toe"]),
        ("Right heel", "right_heel_contact", COLORS["heel"]),
        ("Right forefoot", "right_forefoot_contact", COLORS["forefoot"]),
        ("Right toe", "right_toe_contact", COLORS["toe"]),
    )
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.0
    for row, (label, key, color) in enumerate(rows):
        state = np.asarray(traces[key], dtype=bool)
        start = None
        for index, active in enumerate(np.r_[state, False]):
            if active and start is None:
                start = index
            elif not active and start is not None:
                end = index
                x0 = float(time_s[start])
                width = float(time_s[min(end - 1, len(time_s) - 1)] - x0 + dt)
                ax.broken_barh([(x0, max(width, dt))], (row - 0.34, 0.68), facecolors=color, alpha=0.92)
                start = None
    ax.set_yticks(range(len(rows)), [label for label, _, _ in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()
    ax.set_ylabel("contact lane")
    for row in range(len(rows)):
        ax.axhline(row, color=GRID, linewidth=0.45, alpha=0.7, zorder=0)


def _contact_mechanics(result: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ contact and force-platform mechanics", color=TEXT, fontsize=14, fontweight="bold")

    _phase_context(axes[0], result)
    _contact_raster(axes[0], time_s, traces)
    _style_axis(axes[0], "Contact raster — active intervals", None)
    _add_legend(axes[0], [Patch(facecolor=COLORS["heel"], label="Heel"), Patch(facecolor=COLORS["forefoot"], label="Forefoot"), Patch(facecolor=COLORS["toe"], label="Toe")], "REGION", loc="upper right", ncol=3)

    _phase_context(axes[1], result)
    cop = np.asarray([np.nan if value is None else float(value) for value in traces["cop_x_m"]], dtype=float)
    axes[1].plot(time_s, cop, color=COLORS["net"], linewidth=1.85, zorder=3)
    _style_axis(axes[1], "Force-platform centre of pressure", "CoP x (m)", limits=_limit(scales, "cop_x_m", cop))
    _add_legend(axes[1], [Line2D([], [], color=COLORS["net"], linewidth=2, label="Centre of pressure x")], "CHANNEL")

    _phase_context(axes[2], result, labels=True)
    _plot_model(axes[2], time_s, traces["left_slip_vx_m_s"], color=COLORS["left"], linestyle="-")
    _plot_model(axes[2], time_s, traces["right_slip_vx_m_s"], color=COLORS["right"], linestyle="--")
    _style_axis(axes[2], "Contact slip diagnostics", "speed (m/s)", limits=_limit(scales, "slip_m_s", traces["left_slip_vx_m_s"]))
    _add_legend(axes[2], _side_handles(colors=True), "SIDE", loc="lower right")
    axes[2].set_xlabel("time (s)", color=TEXT)
    _set_time_axis(axes, scales)
    _finish(fig, "contact_mechanics.png")


def _phase_events(result: dict[str, Any], scales: dict[str, tuple[float, float]]) -> None:
    time_s, traces = _time_values(result)
    fig, ax = plt.subplots(1, 1, figsize=(13, 4), constrained_layout=True)
    _phase_context(ax, result, labels=True, strong=True)
    ax.step(time_s, traces["phase_index"], where="post", color=OFF_WHITE, linewidth=2.0, zorder=3)
    _style_axis(ax, "Source CMJ phase dissection and event timing", "phase", limits=(-0.5, len(PHASE_NAMES) - 0.5))
    ax.set_yticks(range(len(PHASE_NAMES)), [name.replace("_", " ") for name in PHASE_NAMES])
    _add_legend(ax, [Line2D([], [], color=EVENT_COLORS["movement_onset_time_s"], linestyle=(0, (3, 2)), label="Movement onset"),
                     Line2D([], [], color=EVENT_COLORS["takeoff_time_s"], linestyle=(0, (3, 2)), label="Takeoff"),
                     Line2D([], [], color=EVENT_COLORS["landing_time_s"], linestyle=(0, (3, 2)), label="Landing")],
                "EVENT", loc="lower right", ncol=3)
    ax.set_xlabel("time (s)", color=TEXT)
    _set_time_axis((ax,), scales)
    _finish(fig, "phase_events.png")


def _format_value(value: Any, *, unit: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.5g}{(' ' + unit) if unit else ''}"


def _metric_block(ax: Any, title: str, rows: list[tuple[str, Any]]) -> None:
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.text(0.03, 0.97, title, transform=ax.transAxes, color=SKY_BLUE, fontsize=10.5, fontweight="bold", va="top")
    y = 0.90
    step = 0.073 if len(rows) <= 11 else 0.066
    for label, value in rows:
        ax.text(0.03, y, label, transform=ax.transAxes, color=MUTED, fontsize=8.2, va="top")
        ax.text(0.97, y, value, transform=ax.transAxes, color=WARM_GOLD, fontsize=8.4, va="top", ha="right")
        y -= step


def _summary_metrics(result: dict[str, Any], trial: dict[str, Any]) -> None:
    """Render human-readable scalar summaries without recomputing metrics."""

    summary = result["summary"]
    diagnostics = result.get("diagnostics", {})
    events = result.get("events", {})
    traces = result["traces"]
    performance = [
        ("Movement onset", _format_value(events.get("movement_onset_time_s"), unit="s")),
        ("Takeoff time", _format_value(events.get("takeoff_time_s"), unit="s")),
        ("Takeoff velocity", _format_value(summary.get("takeoff_velocity_m_s"), unit="m/s")),
        ("Impulse jump height", _format_value(summary.get("jump_height_im_m"), unit="m")),
        ("Airborne duration", _format_value(summary.get("airborne_duration_s"), unit="s")),
        ("Propulsive impulse", _format_value(summary.get("propulsive_impulse_Ns"), unit="N·s")),
        ("Total net impulse", _format_value(summary.get("total_net_impulse_Ns"), unit="N·s")),
        ("Peak concentric force", _format_value(summary.get("peak_concentric_force_N"), unit="N")),
        ("Mean concentric force", _format_value(summary.get("mean_concentric_force_N"), unit="N")),
        ("Bar peak velocity", _format_value(summary.get("bar_peak_velocity_m_s"), unit="m/s")),
        ("Bar displacement range", _format_value(summary.get("bar_displacement_range_m"), unit="m")),
    ]
    auxiliary_force = max(float(diagnostics.get("qfrc_applied_norm_max", 0.0)), float(diagnostics.get("xfrc_applied_norm_max", 0.0)))
    mechanical = [
        ("Peak total GRF", _format_value(np.max(np.asarray(traces["fz_total_N"], dtype=float)), unit="N")),
        ("Maximum trunk pitch", _format_value(diagnostics.get("root_pitch_abs_max_rad"), unit="rad")),
        ("Maximum trunk pitch rate", _format_value(diagnostics.get("root_pitch_rate_abs_max_rad_s"), unit="rad/s")),
        ("Maximum horizontal/root drift", _format_value(diagnostics.get("root_x_abs_max_m"), unit="m")),
        ("Final joint-velocity norm", _format_value(diagnostics.get("final_joint_velocity_norm"), unit="rad/s")),
        ("Applied auxiliary force", _format_value(auxiliary_force, unit="N")),
        ("Actuator clipping", _format_value(diagnostics.get("drive_asymmetry_clip_count"), unit="count")),
        ("Maximum LPT tether force", _format_value(summary.get("max_lpt_tether_force_N"), unit="N")),
        ("Landing rebound", _format_value(diagnostics.get("landing_rebound_m"), unit="m")),
        ("Heel-off time", _format_value(diagnostics.get("heel_off_time_s"), unit="s")),
    ]
    alpha = float(trial.get("drive_asymmetry_alpha", 0.0))
    bilateral = trial.get("bilateral_measurements") if alpha else None
    if bilateral:
        bilateral_rows = [
            ("Drive asymmetry α", _format_value(alpha)),
            ("Left drive scale", _format_value(diagnostics.get("drive_asymmetry_left_scale"))),
            ("Right drive scale", _format_value(diagnostics.get("drive_asymmetry_right_scale"))),
            ("Left braking impulse", _format_value(bilateral.get("left_braking_impulse_Ns"), unit="N·s")),
            ("Right braking impulse", _format_value(bilateral.get("right_braking_impulse_Ns"), unit="N·s")),
            ("Left propulsive impulse", _format_value(bilateral.get("left_propulsive_impulse_Ns"), unit="N·s")),
            ("Right propulsive impulse", _format_value(bilateral.get("right_propulsive_impulse_Ns"), unit="N·s")),
        ]
    else:
        bilateral_rows = [(label, "not applied") for label in (
            "Drive asymmetry α", "Left drive scale", "Right drive scale",
            "Left braking impulse", "Right braking impulse",
            "Left propulsive impulse", "Right propulsive impulse",
        )]

    fig = plt.figure(figsize=(13, 8.5), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 3, left=0.025, right=0.975, bottom=0.07, top=0.88, wspace=0.12)
    _metric_block(fig.add_subplot(grid[0, 0]), "PERFORMANCE & EVENTS", performance)
    _metric_block(fig.add_subplot(grid[0, 1]), "MECHANICAL / NUMERICAL INTEGRITY", mechanical)
    _metric_block(fig.add_subplot(grid[0, 2]), "BILATERAL CONDITION", bilateral_rows)
    fig.suptitle("Loaded CMJ summary metrics", color=TEXT, fontsize=14, fontweight="bold", x=0.025, ha="left")
    fig.text(0.995, 0.018, f"{SCENARIO_ID} | fixed 20 kg | native SI units", ha="right", va="bottom", color=MUTED, fontsize=7)
    fig.savefig(OUTPUT_DIR / "summary_metrics.png", dpi=170, facecolor=BACKGROUND, edgecolor=BACKGROUND)
    plt.close(fig)


def render_scenario(
    trial: dict[str, Any],
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    scale_context: dict[str, tuple[float, float]] | None = None,
) -> tuple[str, ...]:
    """Write the canonical plot family for one already-produced rollout."""

    global OUTPUT_DIR, SCENARIO_ID
    OUTPUT_DIR = Path(output_dir)
    SCENARIO_ID = str(trial.get("trial_id", "scenario"))
    _configure_style()
    observed = trial["observations"]
    scales = scale_context or build_scale_context([{"trial": trial, "result": result}])
    _observable_fit(result, observed, scales)
    _combined_grf_com_lpt(result, observed, scales)
    _force_plate_metrics(result, observed, scales)
    _global_kinematics(result, scales)
    _foot_kinematics(result, scales)
    _joint_kinematics(result, scales)
    _auxiliary_kinematics(result, scales)
    _contact_mechanics(result, scales)
    _phase_events(result, scales)
    _summary_metrics(result, trial)
    return COMMON_PLOT_FAMILY


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the canonical plot family for one public CMJ trial.")
    parser.add_argument("--trial-id", default="20kg_nominal_a")
    parser.add_argument("--split", choices=("identification", "validation"), default="identification")
    parser.add_argument("--output-dir", type=Path, default=MEDIA)
    args = parser.parse_args(argv)
    trial = load_trial(args.trial_id, split=args.split)
    result = simulate_trial(load_named_parameters("synthetic_reference"), trial)
    names = render_scenario(trial, result, args.output_dir)
    print("Wrote:")
    for name in names:
        print(f"  {args.output_dir / name}")


if __name__ == "__main__":
    main()
