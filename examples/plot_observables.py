"""Generate separated scientific plots from one authoritative CMJ rollout.

This module is deliberately a plotting layer.  Every plotted series is read
directly from ``result["traces"]``, ``result["summary"]``, or the public trial
observations.  It does not derive new measurements, resample signals, or run a
second model.  The phase bands use the source phase timing already returned by
the rollout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from loaded_cmj.dataset import load_trial
from loaded_cmj.parameters import load_named_parameters
from loaded_cmj.plant import PHASE_NAMES
from loaded_cmj.simulation import simulate_trial


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"

# High-contrast dark scientific dashboard palette.  The two signal sources are
# intentionally fixed across every figure: observed = light blue, MuJoCo =
# yellow dotted line.  Phase colors are muted so they remain a background cue.
BACKGROUND = "#000000"
PANEL = "#050505"
TEXT = "#e8edf2"
MUTED = "#9aa6b2"
GRID = "#2b333b"
SPINE = "#65717d"
OBSERVED = "#8bdcff"
MUJOCO = "#f4d35e"
MUJOCO_DIM = "#d9bb4b"
EVENT = "#f2f2f2"
PHASE_COLORS = (
    "#355070",  # weighing
    "#6d597a",  # unweighting
    "#9c6644",  # braking
    "#c8962e",  # propulsion
    "#245c8f",  # flight
    "#6941a5",  # landing absorption
    "#27766f",  # stabilization
)
EVENT_NAMES = (
    ("movement_onset_time_s", "onset"),
    ("takeoff_time_s", "takeoff"),
    ("landing_time_s", "landing"),
)


def _configure_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BACKGROUND,
        "savefig.edgecolor": BACKGROUND,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": SPINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.fontsize": 8,
    })


def _style_axis(ax: Any, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, loc="left", color=TEXT, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT)
    ax.tick_params(axis="both", colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.7)
    ax.grid(True, color=GRID, alpha=0.52, linewidth=0.55)
    ax.set_axisbelow(True)


def _legend(
    ax: Any,
    *,
    ncol: int = 1,
    outside: bool = False,
    loc: str = "upper right",
) -> None:
    kwargs = {
        "loc": loc,
        "ncol": ncol,
        "frameon": True,
        "facecolor": BACKGROUND,
        "edgecolor": GRID,
        "framealpha": 0.92,
        "borderpad": 0.45,
    }
    if outside:
        kwargs.update({"bbox_to_anchor": (1.0, 1.0), "loc": "upper left"})
    legend = ax.legend(**kwargs)
    for label in legend.get_texts():
        label.set_color(TEXT)


def _time_values(result: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    traces = result["traces"]
    return [float(value) for value in traces["time_s"]], traces


def _plot_mujoco(
    ax: Any,
    time_s: Iterable[float],
    values: Iterable[float],
    label: str,
    *,
    linestyle: str = "-",
    alpha: float = 0.96,
    linewidth: float = 1.05,
) -> None:
    time = list(time_s)
    series = list(values)
    # Marker frequency is a display choice only; the underlying trace is not
    # decimated or transformed.
    markevery = max(1, len(series) // 95)
    ax.plot(
        time,
        series,
        color=MUJOCO,
        linestyle=linestyle,
        linewidth=linewidth,
        marker="o",
        markersize=2.0,
        markevery=markevery,
        markerfacecolor=MUJOCO,
        markeredgecolor=MUJOCO,
        markeredgewidth=0.25,
        alpha=alpha,
        label=label,
        zorder=3,
    )


def _plot_observed(
    ax: Any,
    time_s: Iterable[float],
    values: Iterable[float],
    label: str,
    *,
    linewidth: float = 1.55,
) -> None:
    ax.plot(
        list(time_s),
        list(values),
        color=OBSERVED,
        linewidth=linewidth,
        linestyle="-",
        label=label,
        zorder=4,
    )


def _shade_phases(
    ax: Any,
    result: dict[str, Any],
    *,
    labels: bool = False,
) -> None:
    """Show the source's hard-coded CMJ phase dissection behind the traces."""

    timing = result.get("phase_timing_s", {})
    for index, phase_name in enumerate(PHASE_NAMES):
        start_end = timing.get(phase_name.lower())
        if not start_end:
            continue
        start, end = (float(start_end[0]), float(start_end[1]))
        ax.axvspan(
            start,
            end,
            color=PHASE_COLORS[index],
            alpha=0.19,
            linewidth=0,
            zorder=0,
        )
        if labels:
            ax.text(
                (start + end) / 2.0,
                0.97,
                phase_name.replace("_", " "),
                transform=ax.get_xaxis_transform(),
                color=TEXT,
                alpha=0.82,
                fontsize=7.5,
                ha="center",
                va="top",
                rotation=90 if "_" in phase_name else 0,
                clip_on=True,
                zorder=1,
            )


def _event_lines(
    ax: Any,
    result: dict[str, Any],
    *,
    labels: bool = False,
) -> None:
    events = result.get("events", {})
    for key, label in EVENT_NAMES:
        value = events.get(key)
        if value is None:
            continue
        time_s = float(value)
        ax.axvline(
            time_s,
            color=EVENT,
            linewidth=0.75,
            linestyle="--",
            alpha=0.62,
            zorder=1,
        )
        if labels:
            ax.text(
                time_s,
                0.06,
                label,
                transform=ax.get_xaxis_transform(),
                color=EVENT,
                alpha=0.86,
                fontsize=7.5,
                ha="left",
                va="bottom",
                rotation=90,
                clip_on=True,
                zorder=2,
            )


def _prepare_axis(ax: Any, result: dict[str, Any], *, labels: bool = False) -> None:
    _shade_phases(ax, result, labels=labels)
    _event_lines(ax, result, labels=labels)


def _finish(fig: Any, filename: str) -> None:
    fig.savefig(MEDIA / filename, dpi=170, facecolor=BACKGROUND, edgecolor=BACKGROUND)
    plt.close(fig)


def _combined_measurements(result: dict[str, Any], observed: dict[str, Any]) -> None:
    """The one intentionally mixed measurement figure: force plates + LPT."""

    time_s, traces = _time_values(result)
    observed_time = [float(value) for value in observed["time_s"]]
    fig, axes = plt.subplots(5, 1, figsize=(13, 13.5), sharex=True, constrained_layout=True)
    fig.suptitle(
        "Loaded CMJ measurements — force plates + bar/LPT",
        color=TEXT,
        fontsize=14,
        fontweight="bold",
    )

    _prepare_axis(axes[0], result)
    _plot_mujoco(axes[0], time_s, traces["fz_left_N"], "MuJoCo left Fz", linestyle="--", alpha=0.72)
    _plot_mujoco(axes[0], time_s, traces["fz_right_N"], "MuJoCo right Fz", linestyle=":", alpha=0.72)
    _plot_mujoco(axes[0], time_s, traces["fz_total_N"], "MuJoCo total Fz")
    _plot_observed(axes[0], observed_time, observed["fz_total_N"], "observed total Fz")
    _style_axis(axes[0], "Bilateral force-platform signal", "force (N)")
    _legend(axes[0], ncol=2)

    _prepare_axis(axes[1], result)
    _plot_mujoco(axes[1], time_s, traces["fnet_N"], "MuJoCo net GRF")
    _style_axis(axes[1], "Net ground-reaction force", "net force (N)")
    _legend(axes[1])

    _prepare_axis(axes[2], result)
    _plot_mujoco(axes[2], time_s, traces["bar_displacement_m"], "MuJoCo bar displacement")
    _plot_observed(axes[2], observed_time, observed["bar_displacement_m"], "observed bar displacement")
    _style_axis(axes[2], "Bar/LPT displacement — bar displacement, not COM displacement", "m")
    _legend(axes[2])

    _prepare_axis(axes[3], result)
    _plot_mujoco(axes[3], time_s, traces["bar_velocity_m_s"], "MuJoCo bar velocity")
    _plot_observed(axes[3], observed_time, observed["bar_velocity_m_s"], "observed bar velocity")
    _style_axis(axes[3], "Bar/LPT velocity", "velocity (m/s)")
    _legend(axes[3], loc="lower right")

    _prepare_axis(axes[4], result, labels=True)
    _plot_mujoco(axes[4], time_s, traces["lpt_tether_force_N"], "MuJoCo LPT tether force")
    _style_axis(axes[4], "LPT tether diagnostic", "force (N)")
    axes[4].set_xlabel("time (s)", color=TEXT)
    _legend(axes[4], loc="lower right")
    for ax in axes:
        ax.set_xlim(time_s[0], time_s[-1])
    _finish(fig, "observable_fit.png")


def _force_plate_metrics(result: dict[str, Any]) -> None:
    time_s, traces = _time_values(result)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11.5), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ force-platform metrics", color=TEXT, fontsize=14, fontweight="bold")

    _prepare_axis(axes[0], result)
    _plot_mujoco(axes[0], time_s, traces["fz_left_N"], "MuJoCo left plate", linestyle="--", alpha=0.72)
    _plot_mujoco(axes[0], time_s, traces["fz_right_N"], "MuJoCo right plate", linestyle=":", alpha=0.72)
    _plot_mujoco(axes[0], time_s, traces["fz_total_N"], "MuJoCo bilateral total")
    _plot_mujoco(axes[0], time_s, traces["total_fz_N"], "MuJoCo raw contact total", linestyle="-.", alpha=0.68)
    _style_axis(axes[0], "Bilateral vertical force plates", "force (N)")
    _legend(axes[0], ncol=3)

    _prepare_axis(axes[1], result)
    for key in (
        "left_heel_fz_N", "left_forefoot_fz_N", "left_toe_fz_N",
        "right_heel_fz_N", "right_forefoot_fz_N", "right_toe_fz_N",
    ):
        _plot_mujoco(axes[1], time_s, traces[key], f"MuJoCo {key.removesuffix('_fz_N')}", alpha=0.78)
    _style_axis(axes[1], "Regional vertical contact forces", "force (N)")
    _legend(axes[1], ncol=3)

    _prepare_axis(axes[2], result)
    for key in (
        "left_heel_fx_N", "left_forefoot_fx_N", "left_toe_fx_N",
        "right_heel_fx_N", "right_forefoot_fx_N", "right_toe_fx_N",
    ):
        _plot_mujoco(axes[2], time_s, traces[key], f"MuJoCo {key.removesuffix('_fx_N')}", alpha=0.78)
    _style_axis(axes[2], "Regional horizontal contact forces", "force (N)")
    _legend(axes[2], ncol=3)

    _prepare_axis(axes[3], result, labels=True)
    _plot_mujoco(axes[3], time_s, traces["total_fx_N"], "MuJoCo total Fx")
    _style_axis(axes[3], "Total horizontal force-plate signal", "force (N)")
    axes[3].set_xlabel("time (s)", color=TEXT)
    _legend(axes[3], loc="lower right")
    for ax in axes:
        ax.set_xlim(time_s[0], time_s[-1])
    _finish(fig, "force_plate_metrics.png")


def _global_kinematics(result: dict[str, Any]) -> None:
    time_s, traces = _time_values(result)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11.5), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ global kinematics", color=TEXT, fontsize=14, fontweight="bold")

    _prepare_axis(axes[0], result)
    _plot_mujoco(axes[0], time_s, traces["root_x_m"], "MuJoCo pelvis/root x", linestyle="--")
    _plot_mujoco(axes[0], time_s, traces["com_x_m"], "MuJoCo COM x")
    _style_axis(axes[0], "Horizontal position", "position (m)")
    _legend(axes[0])

    _prepare_axis(axes[1], result)
    _plot_mujoco(axes[1], time_s, traces["root_z_m"], "MuJoCo pelvis/root z", linestyle="--")
    _plot_mujoco(axes[1], time_s, traces["com_z_m"], "MuJoCo COM z")
    _style_axis(axes[1], "Vertical position", "position (m)")
    _legend(axes[1])

    _prepare_axis(axes[2], result)
    _plot_mujoco(axes[2], time_s, traces["bar_z_m"], "MuJoCo bar z", linestyle="--")
    _plot_mujoco(axes[2], time_s, traces["bar_displacement_m"], "MuJoCo bar displacement")
    _style_axis(axes[2], "Loaded bar position channels", "position (m)")
    _legend(axes[2])

    _prepare_axis(axes[3], result, labels=True)
    _plot_mujoco(axes[3], time_s, traces["root_x_velocity_m_s"], "MuJoCo root x velocity", linestyle="--")
    _plot_mujoco(axes[3], time_s, traces["root_z_velocity_m_s"], "MuJoCo root z velocity", linestyle=":")
    _plot_mujoco(axes[3], time_s, traces["bar_velocity_m_s"], "MuJoCo bar velocity")
    _style_axis(axes[3], "Global velocities", "velocity (m/s)")
    axes[3].set_xlabel("time (s)", color=TEXT)
    _legend(axes[3], ncol=3, loc="lower right")
    for ax in axes:
        ax.set_xlim(time_s[0], time_s[-1])
    _finish(fig, "global_kinematics.png")


def _foot_kinematics(result: dict[str, Any]) -> None:
    time_s, traces = _time_values(result)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ foot kinematics", color=TEXT, fontsize=14, fontweight="bold")

    _prepare_axis(axes[0], result)
    for key in (
        "left_heel_z_m", "left_forefoot_z_m", "left_toe_z_m",
        "right_heel_z_m", "right_forefoot_z_m", "right_toe_z_m",
    ):
        _plot_mujoco(axes[0], time_s, traces[key], f"MuJoCo {key.removesuffix('_z_m')}", alpha=0.78)
    _style_axis(axes[0], "Foot landmark heights", "height (m)")
    _legend(axes[0], ncol=3)

    _prepare_axis(axes[1], result, labels=True)
    _plot_mujoco(axes[1], time_s, traces["foot_clearance_m"], "MuJoCo minimum foot clearance")
    _style_axis(axes[1], "Minimum bilateral foot clearance", "clearance (m)")
    axes[1].set_xlabel("time (s)", color=TEXT)
    _legend(axes[1])
    for ax in axes:
        ax.set_xlim(time_s[0], time_s[-1])
    _finish(fig, "foot_kinematics.png")


def _joint_kinematics(result: dict[str, Any]) -> None:
    time_s, traces = _time_values(result)
    position_groups = (
        (
            "Anatomical joint positions",
            (
                ("left_hip_rad", "hip L"), ("right_hip_rad", "hip R"),
                ("left_knee_rad", "knee L"), ("right_knee_rad", "knee R"),
                ("left_ankle_rad", "ankle L"), ("right_ankle_rad", "ankle R"),
                ("lumbar_pitch_rad", "lumbar"),
                ("left_shoulder_rad", "shoulder L"), ("right_shoulder_rad", "shoulder R"),
                ("left_elbow_rad", "elbow L"), ("right_elbow_rad", "elbow R"),
            ),
        ),
        (
            "Foot and bar-rack joint positions",
            (
                ("left_forefoot_rocker_rad", "rocker L"), ("right_forefoot_rocker_rad", "rocker R"),
                ("left_mtp_rad", "MTP L"), ("right_mtp_rad", "MTP R"),
                ("bar_rack_x_m", "bar rack x"), ("bar_rack_z_m", "bar rack z"),
                ("bar_rack_pitch_rad", "bar rack pitch"),
            ),
        ),
    )
    velocity_groups = (
        (
            "Anatomical joint velocities",
            (
                ("left_hip_velocity_rad_s", "hip L"), ("right_hip_velocity_rad_s", "hip R"),
                ("left_knee_velocity_rad_s", "knee L"), ("right_knee_velocity_rad_s", "knee R"),
                ("left_ankle_velocity_rad_s", "ankle L"), ("right_ankle_velocity_rad_s", "ankle R"),
                ("lumbar_pitch_rate_rad_s", "lumbar"),
                ("left_shoulder_velocity_rad_s", "shoulder L"), ("right_shoulder_velocity_rad_s", "shoulder R"),
                ("left_elbow_velocity_rad_s", "elbow L"), ("right_elbow_velocity_rad_s", "elbow R"),
            ),
        ),
        (
            "Foot and bar-rack joint velocities",
            (
                ("left_forefoot_rocker_velocity_rad_s", "rocker L"),
                ("right_forefoot_rocker_velocity_rad_s", "rocker R"),
                ("left_mtp_velocity_rad_s", "MTP L"), ("right_mtp_velocity_rad_s", "MTP R"),
                ("bar_rack_x_velocity_m_s", "bar rack x"),
                ("bar_rack_z_velocity_m_s", "bar rack z"),
                ("bar_rack_pitch_velocity_rad_s", "bar rack pitch"),
            ),
        ),
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 14.5), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ joint kinematics — q and qdot", color=TEXT, fontsize=14, fontweight="bold")
    groups = position_groups + velocity_groups
    for index, (title, channels) in enumerate(groups):
        _prepare_axis(axes[index], result, labels=index == len(groups) - 1)
        for channel, label in channels:
            _plot_mujoco(axes[index], time_s, traces[channel], f"MuJoCo {label}", alpha=0.78)
        ylabel = "q (rad; bar translations m)" if index < 2 else "qdot (rad/s; bar translations m/s)"
        _style_axis(axes[index], title, ylabel)
        _legend(
            axes[index],
            ncol=4,
            loc="lower right" if index == len(groups) - 1 else "upper right",
        )
    axes[-1].set_xlabel("time (s)", color=TEXT)
    for ax in axes:
        ax.set_xlim(time_s[0], time_s[-1])
    _finish(fig, "joint_kinematics.png")


def _contact_mechanics(result: dict[str, Any]) -> None:
    time_s, traces = _time_values(result)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Loaded CMJ contact and force-platform mechanics", color=TEXT, fontsize=14, fontweight="bold")

    _prepare_axis(axes[0], result)
    for key in (
        "left_heel_contact", "left_forefoot_contact", "left_toe_contact",
        "right_heel_contact", "right_forefoot_contact", "right_toe_contact",
    ):
        _plot_mujoco(axes[0], time_s, [int(value) for value in traces[key]], f"MuJoCo {key.removesuffix('_contact')}", alpha=0.78)
    axes[0].set_ylim(-0.08, 1.08)
    axes[0].set_yticks((0, 1), labels=("off", "on"))
    _style_axis(axes[0], "Per-region contact state", "contact")
    _legend(axes[0], ncol=3)

    _prepare_axis(axes[1], result)
    _plot_mujoco(axes[1], time_s, traces["cop_x_m"], "MuJoCo center of pressure x")
    _style_axis(axes[1], "Force-platform center of pressure", "CoP x (m)")
    _legend(axes[1])

    _prepare_axis(axes[2], result, labels=True)
    _plot_mujoco(axes[2], time_s, traces["left_slip_vx_m_s"], "MuJoCo left slip speed", linestyle="--")
    _plot_mujoco(axes[2], time_s, traces["right_slip_vx_m_s"], "MuJoCo right slip speed")
    _style_axis(axes[2], "Contact slip diagnostics", "speed (m/s)")
    axes[2].set_xlabel("time (s)", color=TEXT)
    _legend(axes[2], loc="lower right")
    for ax in axes:
        ax.set_xlim(time_s[0], time_s[-1])
    _finish(fig, "contact_mechanics.png")


def _phase_events(result: dict[str, Any]) -> None:
    time_s, traces = _time_values(result)
    fig, ax = plt.subplots(1, 1, figsize=(13, 3.5), constrained_layout=True)
    _style_axis(ax, "Source CMJ phase dissection and event timing", "phase index")
    _shade_phases(ax, result, labels=True)
    _event_lines(ax, result, labels=True)
    _plot_mujoco(ax, time_s, traces["phase_index"], "MuJoCo phase index")
    ax.set_yticks(range(len(PHASE_NAMES)), labels=[name.replace("_", " ") for name in PHASE_NAMES])
    ax.set_xlabel("time (s)", color=TEXT)
    ax.set_xlim(time_s[0], time_s[-1])
    _legend(ax, loc="lower right")
    _finish(fig, "phase_events.png")


def _summary_metrics(result: dict[str, Any]) -> None:
    """Render existing scalar summaries/diagnostics without recomputing them."""

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.set_title("Loaded CMJ scalar metrics and event diagnostics", loc="left", color=TEXT, pad=12)

    summary = result["summary"]
    diagnostics = result.get("diagnostics", {})
    events = result.get("events", {})
    lines = [
        ("SUMMARY", ""),
        *[(key, f"{float(value):.6g}") for key, value in summary.items()],
        ("", ""),
        ("EVENTS", ""),
        *[(key, "None" if value is None else f"{float(value):.6g}") for key, value in events.items()
          if key.endswith("_time_s") or key in {"takeoff_velocity_m_s", "jump_height_im_m", "airborne_duration_s"}],
        ("", ""),
        ("DIAGNOSTICS", ""),
        *[(key, f"{float(value):.6g}") for key, value in diagnostics.items()
          if isinstance(value, (int, float)) and key not in {"used_mujoco"}],
    ]
    split = max(1, len(lines) // 2)
    columns = (lines[:split], lines[split:])
    for col, rows in enumerate(columns):
        x = 0.02 + col * 0.49
        y = 0.96
        for key, value in rows:
            if not key:
                y -= 0.018
                continue
            if value == "":
                ax.text(x, y, key, transform=ax.transAxes, color=OBSERVED, fontsize=10, fontweight="bold", va="top")
                y -= 0.038
                continue
            ax.text(x, y, key, transform=ax.transAxes, color=MUTED, fontsize=8.5, va="top")
            ax.text(x + 0.31, y, value, transform=ax.transAxes, color=MUJOCO, fontsize=8.5, va="top")
            y -= 0.028
    # Small source legend, kept as a visual key for the rest of the plot set.
    ax.add_patch(Rectangle((0.02, 0.015), 0.018, 0.012, transform=ax.transAxes, color=OBSERVED, clip_on=False))
    ax.text(0.045, 0.021, "observed = light blue", transform=ax.transAxes, color=OBSERVED, fontsize=8.5, va="center")
    ax.add_patch(Rectangle((0.22, 0.015), 0.018, 0.012, transform=ax.transAxes, color=MUJOCO, clip_on=False))
    ax.text(0.245, 0.021, "MuJoCo = yellow dotted", transform=ax.transAxes, color=MUJOCO, fontsize=8.5, va="center")
    _finish(fig, "summary_metrics.png")


def main() -> None:
    _configure_style()
    MEDIA.mkdir(parents=True, exist_ok=True)
    trial = load_trial("public_001", split="identification")
    # One rollout feeds every figure.  No plot invokes simulation a second time.
    result = simulate_trial(load_named_parameters("synthetic_reference"), trial)
    observed = trial["observations"]
    _combined_measurements(result, observed)
    _force_plate_metrics(result)
    _global_kinematics(result)
    _foot_kinematics(result)
    _joint_kinematics(result)
    _contact_mechanics(result)
    _phase_events(result)
    _summary_metrics(result)
    print("Wrote:")
    for name in (
        "observable_fit.png",
        "force_plate_metrics.png",
        "global_kinematics.png",
        "foot_kinematics.png",
        "joint_kinematics.png",
        "contact_mechanics.png",
        "phase_events.png",
        "summary_metrics.png",
    ):
        print(f"  {MEDIA / name}")


if __name__ == "__main__":
    main()
