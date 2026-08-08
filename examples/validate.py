"""Validate the synthetic reference configuration on public validation trials."""

from __future__ import annotations

import json

from loaded_cmj.dataset import load_trials
from loaded_cmj.parameters import load_named_parameters
from loaded_cmj.validation import validate_parameters


def main() -> None:
    report = validate_parameters(
        load_named_parameters("synthetic_reference"),
        load_trials("validation"),
        max_trials=2,
    )
    compact = {
        "valid": report["valid"],
        "trial_count": report["trial_count"],
        "aggregate": report["aggregate"],
        "trial_ids": [trial["trial_id"] for trial in report["trials"]],
    }
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

