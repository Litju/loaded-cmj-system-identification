"""Write the fixed-load bilateral force qualification figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from loaded_cmj.dataset import load_trial
import plot_observables as plotting


ROOT = Path(__file__).resolve().parents[1]


def _event_time(trial: dict[str, object], key: str) -> float:
    value = trial["observed_events"].get(key)
    if value is None:
        raise ValueError(f"missing {key} in {trial['trial_id']}")
    return float(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the physical bilateral force qualification figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "media" / "20kg_bilateral_asymmetry" / "bilateral_force_asymmetry.png",
    )
    args = parser.parse_args(argv)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    nominal = load_trial("20kg_nominal_a", split="identification")
    asymmetry = load_trial("20kg_bilateral_asymmetry", split="identification")
    nominal_obs = nominal["observations"]
    asym_obs = asymmetry["observations"]
    time_s = np.asarray(asym_obs["time_s"], dtype=float)

    fig, (force_ax, impulse_ax) = plt.subplots(
        2, 1, figsize=(13, 7.4), sharex=False,
        gridspec_kw={"height_ratios": [3.2, 1.25]}, constrained_layout=True,
    )
    fig.patch.set_facecolor(plotting.BACKGROUND)
    for ax in (force_ax, impulse_ax):
        plotting._style_axis(ax, "")

    left = np.asarray(asym_obs["fz_left_N"], dtype=float)
    right = np.asarray(asym_obs["fz_right_N"], dtype=float)
    nominal_left = np.asarray(nominal_obs["fz_left_N"], dtype=float)
    nominal_right = np.asarray(nominal_obs["fz_right_N"], dtype=float)
    nominal_time = np.asarray(nominal_obs["time_s"], dtype=float)
    markevery = max(1, len(time_s) // 18)
    force_ax.plot(
        time_s, left, color=plotting.COLORS["left"], linewidth=2.0,
        marker="o", markevery=markevery, markersize=3.2,
        markerfacecolor=plotting.BACKGROUND, markeredgecolor=plotting.COLORS["left"],
        markeredgewidth=0.8, label="asymmetry",
    )
    force_ax.plot(
        time_s, right, color=plotting.COLORS["right"], linewidth=2.0,
        marker="s", markevery=markevery, markersize=3.0,
        markerfacecolor=plotting.BACKGROUND, markeredgecolor=plotting.COLORS["right"],
        markeredgewidth=0.8,
    )
    force_ax.plot(nominal_time, nominal_left, color=plotting.COLORS["left"], linewidth=1.15, linestyle="--", alpha=0.62)
    force_ax.plot(nominal_time, nominal_right, color=plotting.COLORS["right"], linewidth=1.15, linestyle="--", alpha=0.62)

    onset = _event_time(asymmetry, "movement_onset_time_s")
    takeoff = _event_time(asymmetry, "takeoff_time_s")
    landing_value = asymmetry["observed_events"].get("landing_time_s")
    landing = float(landing_value) if landing_value is not None else 2.58
    force_ax.axvspan(0.0, onset, color=plotting.PHASE_COLORS[0], alpha=0.06, linewidth=0)
    force_ax.axvspan(onset, takeoff, color=plotting.PHASE_COLORS[2], alpha=0.06, linewidth=0)
    force_ax.axvspan(takeoff, landing, color=plotting.PHASE_COLORS[4], alpha=0.06, linewidth=0)
    force_ax.axvline(onset, color=plotting.EVENT_COLORS["movement_onset_time_s"], linewidth=0.95, linestyle=(0, (3, 2)))
    force_ax.axvline(takeoff, color=plotting.EVENT_COLORS["takeoff_time_s"], linewidth=0.95, linestyle=(0, (3, 2)))
    force_ax.axvline(landing, color=plotting.EVENT_COLORS["landing_time_s"], linewidth=0.95, linestyle=(0, (3, 2)))
    for event_time, label, color in (
        (onset, "onset", plotting.EVENT_COLORS["movement_onset_time_s"]),
        (takeoff, "takeoff", plotting.EVENT_COLORS["takeoff_time_s"]),
        (landing, "landing", plotting.EVENT_COLORS["landing_time_s"]),
    ):
        force_ax.text(event_time, 0.04, label, transform=force_ax.get_xaxis_transform(), color=color,
                      fontsize=8, va="bottom", ha="left", rotation=90)
    force_ax.set_title("Fixed 20 kg loaded CMJ — bilateral force-platform qualification (α = 0.02)",
                       color=plotting.TEXT, fontsize=13, loc="left", pad=10)
    force_ax.set_ylabel("vertical force (N)", color=plotting.TEXT)
    force_ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    force_ax.set_ylim(*plotting._limits(np.r_[left, right, nominal_left, nominal_right]))
    plotting._add_legend(force_ax, [
        Line2D([], [], color=plotting.COLORS["left"], linewidth=2, marker="o", markersize=4,
               markerfacecolor=plotting.BACKGROUND, label="Left"),
        Line2D([], [], color=plotting.COLORS["right"], linewidth=2, marker="s", markersize=4,
               markerfacecolor=plotting.BACKGROUND, label="Right"),
    ], "SIDE")
    plotting._add_legend(force_ax, [
        Line2D([], [], color=plotting.TEXT, linewidth=2, label="α = 0.02 synthetic condition"),
        Line2D([], [], color=plotting.TEXT, linewidth=1.1, linestyle="--", alpha=0.65, label="Nominal reference"),
    ], "CONDITION", loc="upper right")

    bilateral = asymmetry["bilateral_measurements"]
    labels = ["Braking", "Propulsive"]
    left_impulses = [bilateral["left_braking_impulse_Ns"], bilateral["left_propulsive_impulse_Ns"]]
    right_impulses = [bilateral["right_braking_impulse_Ns"], bilateral["right_propulsive_impulse_Ns"]]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    impulse_ax.bar(x - width / 2, left_impulses, width, color=plotting.COLORS["left"], label="left")
    impulse_ax.bar(x + width / 2, right_impulses, width, color=plotting.COLORS["right"], label="right")
    impulse_ax.set_xticks(x, labels)
    impulse_ax.set_ylabel("impulse (N·s)", color=plotting.TEXT)
    impulse_ax.set_title("Direct side-specific event-window impulses; no normalized clinical index",
                         color=plotting.TEXT, fontsize=10, loc="left", pad=7)
    impulse_ax.legend(frameon=False, fontsize=8.5, labelcolor=plotting.TEXT, loc="upper left")
    impulse_ax.set_xlabel("phase window", color=plotting.TEXT)

    fig.text(0.995, 0.006, "20kg_bilateral_asymmetry | fixed 20 kg | direct synthetic measurements",
             ha="right", va="bottom", color=plotting.MUTED, fontsize=7)
    fig.savefig(output, dpi=180, facecolor=plotting.BACKGROUND, edgecolor=plotting.BACKGROUND)
    plt.close(fig)
    print(json.dumps({"output": str(output), "resolution": "1980x1224", "alpha": 0.02}))


if __name__ == "__main__":
    main()
