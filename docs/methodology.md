# Methodology

## Forward experiment

The experiment uses deterministic 2 ms MuJoCo rollouts of the committed
loaded-CMJ MJCF. A trial specifies external bar load, countermovement-depth and
phase-duration scales, and the source simulation duration. The plant performs a
2 s settling rollout, resets time, then records the complete trial sequence.
The settling interval is an initialization procedure; it is not included in the
reported trial time series.

The source-defined CMJ dissection is weighing, unweighting, braking,
propulsion, flight, landing absorption, and stabilization. The controller's
phase path and the event detector are both retained, but measured contact
events remain the authority for takeoff and landing.

## Observation path

The observation pipeline:

1. runs the full plant at its native timestep;
2. records bilateral force-platform, bar/LPT, contact, root, COM, joint, and
   bar-rack telemetry;
3. applies the source force-platform weighing/net-force transform;
4. preserves LPT delay, filtering, scaling, offset, and bar-only
   interpretation;
5. compares observations on the declared 100 Hz grid.

The native trace and comparison observation are different representations of
the same trial. Resampling does not alter the plant state or replace the
measurement model.

## Bounded fitting

`loaded_cmj.identification.identify_parameters` varies explicitly named
coordinates of the complete 27-scalar parameter vector within the bounds in
`configs/param_schema.json`. For each residual evaluation it:

```text
parameter coordinates
    -> full MuJoCo model construction
    -> deterministic forward trial rollouts
    -> force/bar observable extraction
    -> unit-balanced trace and summary residuals
    -> bounded nonlinear least-squares update
```

Force residuals are scaled by the observed quiet baseline and bar-displacement
residuals by the declared 0.1 m fitting scale; event/summary residuals use the
source explicit scales in `observation_residual_vector`. The result retains the
full structured parameter object and reports optimizer status and rollout
failures.

## Validation

Validation is separate from fitting. `validate_trial` and
`validate_parameters` report raw-unit force/bar errors plus finite-state,
contact, event, phase, flight, countermovement, landing, posture, tether, and
auxiliary-force diagnostics. No single normalized score is used as a scientific
substitute for the traces or mechanics.

See [`docs/measurements.md`](measurements.md) for channel semantics,
[`docs/identification.md`](identification.md) for fitting use, and
[`docs/validation.md`](validation.md) for validity gates.
