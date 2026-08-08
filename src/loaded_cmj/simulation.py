"""Public simulation wrapper around the authoritative MuJoCo plant."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import plant


def trial_descriptor(trial: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a dataset trial into the plant's simulation descriptor.

    Dataset files call the external load ``bar_load_kg`` because it is the
    measured bar condition. The dynamics API uses ``external_load_kg``; both
    refer to the same physical load and no COM quantity is substituted.
    """

    descriptor = deepcopy(trial) if trial is not None else {}
    if "external_load_kg" not in descriptor and "bar_load_kg" in descriptor:
        descriptor["external_load_kg"] = float(descriptor["bar_load_kg"])
    descriptor.pop("observations", None)
    descriptor.pop("observed_events", None)
    descriptor.pop("observed_summary", None)
    return plant._default_trial(descriptor)


def simulate_trial(
    params: dict[str, Any] | None = None,
    trial: dict[str, Any] | None = None,
    *,
    record: bool = True,
) -> dict[str, Any]:
    """Run one deterministic full-morphology loaded-CMJ rollout."""

    active_params = plant.default_params() if params is None else deepcopy(params)
    return plant.run_trial(active_params, trial_descriptor(trial), record=record)

