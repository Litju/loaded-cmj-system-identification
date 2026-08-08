# Methodology

## Scope and design

This release is a deterministic within-plant system-identification study at a
fixed 20 kg external bar load. Every identification and validation fixture
uses the same load while varying only the validated depth, phase-timing,
measurement-realization, and qualified known-excitation conditions documented
in the dataset manifest. The modeled athlete is a synthetic rigid-body
multibody abstraction; it is not a muscle-level physiological model or a
subject-specific digital twin.

The separation between the plant, controller, and sampled measurement route is
methodologically informed by `@astrom_murray_2008`, `@sarkka_svensson_2023`,
and `@featherstone_2008`; those sources do not provide the numerical plant
parameters.

The two nominal conditions have identical mechanics and distinct declared
measurement/noise seeds. The bilateral condition applies a balanced,
experiment-layer drive scaling with the smallest qualified alpha and reports
the direct left/right force and impulse traces. Alpha is known and fixed; it is
not part of the fitted parameter vector. Because these are deterministic
synthetic conditions, effect sizes and replay errors are reported rather than
population-inference p-values.

## Forward experiment

The experiment uses deterministic 2 ms MuJoCo rollouts of the committed
loaded-CMJ MJCF. A trial specifies the fixed 20 kg external bar load,
countermovement-depth and phase-duration scales, and the source simulation
duration. The plant performs a
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
2. records left/right/total force-platform, bar/LPT, contact, root, COM, joint, and
   bar-rack telemetry;
3. applies the source force-platform weighing/net-force transform;
4. preserves LPT delay, filtering, scaling, offset, and bar-only
   interpretation;
5. compares observations on the declared 100 Hz grid.

The native trace and comparison observation are different representations of
the same trial. Resampling does not alter the plant state or replace the
measurement model.

Force-platform and impulse terminology follows the measurement-method boundary
recorded in `@hamill_knutzen_derrick_2015` and `@zatsiorsky_kinetics_2002`.

## Units and uncertainty

The public signal surface uses SI units: force in N, displacement in m,
velocity in m/s, impulse in N·s, time in s, mass in kg, and angles in rad.
`configs/param_schema.json` supplies units for every fitted coordinate. The
model gravity is `g = 9.81 m/s²`; the reported impulse-momentum height is exactly
`hIM = v_takeoff² / (2g)`, not an optical or flight-height measurement.

The generator's force and bar-noise standard deviations define deterministic
measurement realizations selected by recorded seeds. They are not empirical
instrument uncertainty estimates, confidence intervals, or population error
bars. The release reports deterministic replay error and physical
left/right differences; it makes no inferential claim from the six synthetic
conditions.

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

The experiment-design, model-structure, and bounded-optimization framing is
literature-informed (`@ljung_1999`, `@beck_1979`, and
`@nocedal_wright_2006`). The 27 coordinates, bounds, residual scales, and
optimizer budget remain first-party choices.

Force residuals are scaled by the observed quiet baseline and bar-displacement
residuals by the declared 0.1 m fitting scale; event/summary residuals use the
source explicit scales in `observation_residual_vector`. The result retains the
full structured parameter object and reports optimizer status and rollout
failures.

The fitted vector contains no bilateral-drive excitation coordinate. The
asymmetry condition is a known trial descriptor and remains fixed during
identification.

## Validation

Validation is separate from fitting. `validate_trial` and
`validate_parameters` report raw-unit force/bar errors plus finite-state,
contact, event, phase, flight, countermovement, landing, posture, tether, and
auxiliary-force diagnostics. No single normalized score is used as a scientific
substitute for the traces or mechanics.

See [`docs/measurements.md`](measurements.md) for channel semantics,
[`docs/identification.md`](identification.md) for fitting use, and
[`docs/validation.md`](validation.md) for validity gates.
