"""Write the fixed-load bilateral force qualification figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loaded_cmj.dataset import load_trial


ROOT = Path(__file__).resolve().parents[1]


def _event_time(trial: dict, key: str) -> float:
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
        2,
        1,
        figsize=(11.0, 6.8),
        sharex=False,
        gridspec_kw={"height_ratios": [3.2, 1.25]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#10141c")
    for ax in (force_ax, impulse_ax):
        ax.set_facecolor("#10141c")
        ax.tick_params(colors="#d7dde8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#475263")
        ax.grid(True, color="#2a3342", linewidth=0.65, alpha=0.7)

    left = np.asarray(asym_obs["fz_left_N"], dtype=float)
    right = np.asarray(asym_obs["fz_right_N"], dtype=float)
    nominal_left = np.asarray(nominal_obs["fz_left_N"], dtype=float)
    nominal_right = np.asarray(nominal_obs["fz_right_N"], dtype=float)
    force_ax.plot(time_s, left, color="#55c7e8", linewidth=1.8, label="asymmetric left GRF")
    force_ax.plot(time_s, right, color="#f2a65a", linewidth=1.8, label="asymmetric right GRF")
    force_ax.plot(time_s, nominal_left, color="#55c7e8", linewidth=0.9, linestyle="--", alpha=0.6, label="nominal left GRF")
    force_ax.plot(time_s, nominal_right, color="#f2a65a", linewidth=0.9, linestyle="--", alpha=0.6, label="nominal right GRF")

    onset = _event_time(asymmetry, "movement_onset_time_s")
    takeoff = _event_time(asymmetry, "takeoff_time_s")
    landing = float(asymmetry["observed_events"].get("landing_time_s", 2.58))
    force_ax.axvspan(0.0, onset, color="#667085", alpha=0.08)
    force_ax.axvspan(onset, takeoff, color="#e9c46a", alpha=0.08)
    force_ax.axvspan(takeoff, landing, color="#6bc7a5", alpha=0.08)
    for event_time, label, color in (
        (onset, "onset", "#e9c46a"),
        (takeoff, "takeoff", "#6bc7a5"),
        (landing, "landing", "#e76f51"),
    ):
        force_ax.axvline(event_time, color=color, linewidth=1.0, alpha=0.9)
        force_ax.text(event_time + 0.018, 0.96, label, transform=force_ax.get_xaxis_transform(), color=color, fontsize=9, va="top")
    force_ax.set_title(
        "Fixed 20 kg loaded CMJ — bilateral force-platform qualification (alpha = 0.02)",
        color="#f3f5f7",
        fontsize=13,
        loc="left",
        pad=10,
    )
    force_ax.set_ylabel("Vertical force (N)", color="#d7dde8")
    force_ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.5, labelcolor="#d7dde8")
    force_ax.set_xlim(float(time_s[0]), float(time_s[-1]))

    bilateral = asymmetry["bilateral_measurements"]
    labels = ["braking", "propulsive"]
    left_impulses = [bilateral["left_braking_impulse_Ns"], bilateral["left_propulsive_impulse_Ns"]]
    right_impulses = [bilateral["right_braking_impulse_Ns"], bilateral["right_propulsive_impulse_Ns"]]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    impulse_ax.bar(x - width / 2, left_impulses, width, color="#55c7e8", label="left")
    impulse_ax.bar(x + width / 2, right_impulses, width, color="#f2a65a", label="right")
    impulse_ax.set_xticks(x, labels)
    impulse_ax.set_ylabel("Impulse (N·s)", color="#d7dde8")
    impulse_ax.set_title(
        "Direct side-specific event-window impulses; no normalized clinical index",
        color="#d7dde8",
        fontsize=10,
        loc="left",
        pad=7,
    )
    impulse_ax.legend(frameon=False, fontsize=8.5, labelcolor="#d7dde8", loc="upper left")

    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({"output": str(output), "resolution": "1980x1224", "alpha": 0.02}))


if __name__ == "__main__":
    main()
