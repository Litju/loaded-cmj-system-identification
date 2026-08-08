# Identification

`loaded_cmj.identification.identify_parameters` is a bounded nonlinear
least-squares research path against bilateral force-platform total and
bar/LPT
displacement observations, with summary and event-derived residual diagnostics.
Every residual evaluation re-runs the source full-morphology MuJoCo plant; no
trajectory is replayed from the observations.

The general model-structure, experiment-design, residual, and identifiability
framing is literature-informed by `@ljung_1999`, `@beck_1979`, and
`@nocedal_wright_2006`. The current 27-coordinate schema, bounds, residual
weights, and deterministic optimizer settings are first-party choices.

## Parameter authority

The canonical authority is `configs/param_schema.json`. It defines the full
27-scalar configuration, including body mass, joint stiffness/damping,
phase-controller gains and activation timing, bar/rack and grip compliance,
contact compliance, force-platform calibration, encoder behavior, bar
attachment, and LPT tether terms. Array coordinates use explicit names such as
`joint_stiffness_Nm_rad[1]` and retain the schema order.

`nominal.json`, `example.json`, `public_reference_fit.json`, and
`synthetic_reference.json` are named configurations, not interchangeable
parameter authorities. The synthetic reference is documented as the
data-generating configuration for the reproducible experiment.

## Fitting result

The fitting result contains:

- the complete structured parameter configuration;
- the varied coordinate names and their bounded optimizer coordinates;
- optimizer success/message, cost, residual norm, and evaluation counts;
- the number of rollouts that left the valid mechanics region.

Coordinates not selected for a fit stay at the supplied initial configuration.
This makes identifiability choices explicit instead of implying that every
parameter is independently recoverable from every trial.

The synthetic reference configuration is useful for reproducing observations;
it does not establish experimental validity or subject-specific inference.

All public fits use the fixed 20 kg experiment. The bilateral drive condition
is a known synthetic excitation and is intentionally absent from the 27-scalar
identification vector; the physical left/right traces remain available for
diagnostics.

## Interpreting the examples

`examples/identify.py` is intentionally a fast smoke example: it uses the
first two identification trials, four named coordinates, and two function
evaluations. Because that cap is deliberately minimal, its printed optimizer
status may be `success: false` with a maximum-function-evaluations message even
when all rollouts are finite; that is expected smoke-run behavior, not a
qualified identification result. A substantive run should state its trial
split, selected coordinates, initial configuration, optimizer budget, residual
sampling, and validation result. The public function defaults to the complete
identification split and a broader explicit coordinate list.
