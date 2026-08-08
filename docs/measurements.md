# Measurements

The measurement path is part of the scientific model, not a rendering helper.
Its public channel names, units, and transformations are defined by
`src/loaded_cmj/plant.py`, `src/loaded_cmj/measurements.py`, and
`configs/preprocessing.json`.

The interpretation of force-platform signals, external/internal force, and
impulse is literature-informed by `@hamill_knutzen_derrick_2015` and
`@zatsiorsky_kinetics_2002`. The sampled-motion terminology is informed by
`@zatsiorsky_kinematics_1998`; all channel transforms and constants below are
first-party implementation choices.

## Force-platform channels

| Channel | Unit | Meaning |
| --- | --- | --- |
| `fz_left_N` | N | Vertical force attributed to the left force-plate region |
| `fz_right_N` | N | Vertical force attributed to the right force-plate region |
| `fz_total_N` | N | Bilateral total vertical force, exactly `fz_left_N + fz_right_N` |
| `fnet_N` | N | `fz_total_N - Wsys`, after the source weighing baseline |
| `total_fz_N` / `total_fx_N` | N | Raw contact-force diagnostics before the canonical measurement transform |
| `cop_x_m` | m | Force-platform center of pressure; undefined outside contact uses the source sentinel semantics |

The weighing baseline `Wsys` is the mean total vertical force over the final
1.00 s of the 1.50 s weighing interval. The source system-mass estimate is
`Wsys / 9.81`; this is a measurement-derived diagnostic, not a replacement for
the model's configured masses.

For the qualified bilateral excitation, left/right propulsive and braking
impulses are descriptive physical quantities. Each is the time integral of the
corresponding side's force after subtracting that side's quiet weighing mean,
over the event-defined concentric or braking interval. They are reported in
N·s in the asymmetry trial's `bilateral_measurements` section and are not
compressed into a clinical or readiness index.

## Bar/LPT channels

| Channel | Unit | Meaning |
| --- | --- | --- |
| `bar_z_m` | m | MuJoCo bar-site vertical position |
| `bar_displacement_m` | m | Bar displacement at the LPT site after the source offset/scale/delay/filter path |
| `bar_velocity_m_s` | m/s | Bar/LPT velocity associated with the source bar-displacement signal |
| `lpt_tether_force_N` | N | Retained diagnostic force for the LPT tether model |

`bar_displacement_m` is bar displacement. It is not center-of-mass (COM)
displacement, and the API, fitting residuals, validation reports, plots, and
renderer labels preserve that distinction. COM traces are separate read-only
MuJoCo state channels.

## Preprocessing and timing

The source path preserves sensor delay, filtering, scale, offset, force
baseline, and channel mapping. The comparison grid is 0.00–3.60 s inclusive at
0.01 s spacing (100 Hz, 361 samples). Linear interpolation onto that grid is
defined in `configs/preprocessing.json` and `loaded_cmj.preprocessing.resample`.

Native telemetry remains available at the 0.002 s plant timestep for mechanics,
plots, and source-equivalence audits. The public `preprocess_observations`
function applies the source weighing/net-force transform and supplies a missing
observed bar velocity from the observation channel convention when necessary.

## Kinematics available for plotting

The same rollout exposes COM position, root state, anatomical joint positions
and velocities, foot landmarks/contact, bar-rack coordinates, center of
pressure, and contact slip diagnostics. These are read-only state/telemetry
channels; the static plotting example only routes existing fields to separate
figures. See [`docs/visualization.md`](visualization.md).
