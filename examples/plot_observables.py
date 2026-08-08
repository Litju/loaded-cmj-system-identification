"""Plot public force-platform and bar/LPT observations against a rollout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from loaded_cmj.dataset import load_trial
from loaded_cmj.parameters import load_named_parameters
from loaded_cmj.simulation import simulate_trial


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    trial = load_trial("public_001", split="identification")
    result = simulate_trial(load_named_parameters("synthetic_reference"), trial)
    observed = trial["observations"]
    predicted = result["traces"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(observed["time_s"], observed["fz_total_N"], color="#202020", lw=1.0, label="observed total Fz")
    axes[0].plot(predicted["time_s"], predicted["fz_total_N"], color="#007f9e", lw=1.2, label="MuJoCo total Fz")
    axes[0].set_ylabel("force (N)")
    axes[0].set_title("Bilateral force-platform measurement")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.22)
    axes[1].plot(observed["time_s"], observed["bar_displacement_m"], color="#202020", lw=1.0, label="observed bar displacement")
    axes[1].plot(predicted["time_s"], predicted["bar_displacement_m"], color="#c85a00", lw=1.2, label="MuJoCo bar displacement")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("bar displacement (m)")
    axes[1].set_title("Bar/LPT measurement (bar displacement, not COM displacement)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.22)
    for ax in axes:
        ax.set_xlim(float(np.min(observed["time_s"])), float(np.max(observed["time_s"])))
    fig.savefig(ROOT / "media" / "observable_fit.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

