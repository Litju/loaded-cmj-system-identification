"""Fit a small, explicit bounded coordinate set on public trials."""

from __future__ import annotations

import json
from pathlib import Path

from loaded_cmj.dataset import load_trials
from loaded_cmj.identification import identify_parameters


def main() -> None:
    trials = load_trials("identification")[:2]
    result = identify_parameters(
        trials,
        parameter_names=(
            "body_mass_kg",
            "braking_gain",
            "propulsive_gain",
            "encoder_offset_m",
        ),
        max_nfev=2,
        residual_sample_count=20,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "parameters"}, indent=2))
    print(json.dumps({"fitted_parameters": result["parameters"]}, indent=2))


if __name__ == "__main__":
    main()

