"""Run one full loaded-CMJ forward simulation and print its diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from loaded_cmj.dataset import load_trial
from loaded_cmj.parameters import load_parameters
from loaded_cmj.simulation import simulate_trial


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    params = load_parameters(ROOT / "configs" / "synthetic_reference.json")
    trial = load_trial("public_001", split="identification")
    result = simulate_trial(params, trial)
    print(json.dumps({
        "trial_id": trial["trial_id"],
        "valid": result["valid"],
        "events": result["events"],
        "summary": result["summary"],
        "diagnostics": result["diagnostics"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

