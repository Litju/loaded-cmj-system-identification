# Validation

Validation compares forward predictions with explicit public observations in
physical units. It is separate from fitting and is intended to expose where a
parameter configuration agrees or disagrees with the measurement and mechanics
contracts.

## Reported quantities

`validate_trial` reports:

- force-platform total RMSE and maximum absolute error;
- bar displacement RMSE and maximum absolute error;
- bar/LPT velocity RMSE;
- errors in available takeoff velocity, impulse-momentum height, propulsive
  impulse, bar range, and bar peak velocity summaries;
- movement-onset, takeoff, and landing timing errors;
- finite-state, contact, phase-order, countermovement, flight, landing,
  posture, tether, and auxiliary-force diagnostics.

`validate_parameters` applies that report across the requested validation
trials and returns aggregate mean/max force and bar-displacement RMSE values.
The raw report remains available under each trial entry; no single scalar is
used as a scientific substitute for the traces or mechanics.

## Event interpretation

The source event interpretation is retained: movement onset follows the
weighing rule, takeoff requires a sustained bilateral no-contact interval after
propulsion, and landing requires sustained recontact. A missing physical
takeoff is represented as `None`; it is never replaced by a scheduled phase
time. The same rule is used by validation, plots, and rendering.

## Running validation

The example is intentionally short:

```bash
python examples/validate.py
```

It validates two trials for quick feedback. To validate the complete checked-in
split:

```python
from loaded_cmj.dataset import load_trials
from loaded_cmj.parameters import load_named_parameters
from loaded_cmj.validation import validate_parameters

report = validate_parameters(
    load_named_parameters("synthetic_reference"),
    load_trials("validation"),
)
assert report["valid"]
```

The thresholds and event definitions are explicitly recorded in
`configs/preprocessing.json` and `src/loaded_cmj/validation.py`.
