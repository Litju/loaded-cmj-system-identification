# Reproducibility and qualification

This checkout is a local owner-review release candidate. It has no hosted
repository, remote URL, DOI, or published release yet. Relative links in this
documentation point to files in this checkout. The code and data can be
released after the owner supplies publication identity metadata; the scientific
qualification below is independent of hosting.

## Fixed experiment contract

The reproducible path is fixed by the following tracked inputs:

- `assets/loaded_cmj_model.xml` and `src/loaded_cmj/plant.py`: source plant,
  contacts, controls, telemetry, events, and timestep;
- `configs/param_schema.json`: parameter names, units, bounds, and ordering;
- `configs/synthetic_reference.json`: the documented synthetic data-generating
  configuration;
- `configs/preprocessing.json`: comparison grid, weighing rule, event rules,
  measurement interpretation, and mechanics thresholds;
- `data/identification_trials.json` and `data/validation_trials.json`: explicit
  public experiment fixtures;
- `tools/generate_dataset.py`: deterministic fixture-generation implementation.

The generator records SHA-256 hashes in `data/dataset_manifest.json`. Generate
into a scratch directory for an audit:

```bash
python tools/generate_dataset.py --output-dir /tmp/loaded-cmj-generated
```

Do not use the repository itself as the output directory unless intentionally
regenerating and reviewing the tracked fixtures.

## Fresh-environment run

From the repository root:

```bash
python -m pip install -e ".[dev]"
pytest
python examples/simulate.py
python examples/identify.py
python examples/validate.py
python tools/generate_dataset.py --output-dir /tmp/loaded-cmj-generated
python examples/plot_observables.py
python examples/render.py
```

The first four commands exercise installation, tests, forward simulation,
bounded fitting, and validation. The generator, plotting script, and renderer
then regenerate the research artifacts without requiring a remote service.
Rendering additionally requires an `ffmpeg` executable and a working MuJoCo
OpenGL backend. Headless systems may need the platform's documented EGL or
OSMesa configuration.

## Recorded migration qualification

The local source-to-target qualification run for this checkout established the
following contract:

| Check | Result |
| --- | --- |
| Compiled model counts | `nq=21`, `nv=21`, `nu=6`, `nbody=19`, `njnt=21`, `ngeom=28`, `nsite=26`, `ntendon=4`, `nsensor=10`, `npair=6`, `neq=0` |
| Source/target plant | Same named bodies, joints, geoms, sites, actuators, sensors, tendons, solver settings, and critical physical arrays |
| Public source-vs-target rollout audit | Six identification trials; maximum qpos/qvel, bilateral force, bar/LPT, and event-time differences were `0.0` in the recorded deterministic run; contact states were equal |
| Validation fixtures | 32 public validation trials passed the finite-state, mechanics, contact/event, and measurement validity path |
| Test suite | 16 tests pass without a source checkout; the optional source-equivalence test makes 17 tests pass when `LOADED_CMJ_SOURCE_ROOT` is supplied |
| Render artifact | 1280×720 MP4, 336 frames, 30 fps, 11.2 s, generated from the target plant with the MakeHuman visual family |

These are source-port and numerical reproducibility results. They are not
experimental human-validation results and do not establish physiological
validity.

## Re-run source equivalence

The source checkout is deliberately external to this repository. Set its root
at runtime and run the read-only harness:

```bash
export LOADED_CMJ_SOURCE_ROOT=/path/to/source/loaded-cmj-forceplate-lpt-sysid
python tools/equivalence_harness.py \
  --source-root "$LOADED_CMJ_SOURCE_ROOT" \
  --trial-id public_001 \
  --tolerance 1e-12
LOADED_CMJ_SOURCE_ROOT="$LOADED_CMJ_SOURCE_ROOT" pytest -q
```

The harness compares compiled-model structure and critical arrays, qpos/qvel,
bilateral force-platform channels, bar/LPT channels, contact states, and event
timing. It does not copy source files, write into the source checkout, or place
comparison dumps in the tracked tree. Repeat the command for
`public_002` through `public_006` for the complete public-trial audit.

Without `LOADED_CMJ_SOURCE_ROOT`, the source-equivalence test is skipped by
design; that is an unavailable external evidence source, not a passing
equivalence claim.

## Numerical and platform notes

- MuJoCo is constrained by `pyproject.toml` to `>=3.2,<4`.
- The physical rollout timestep is 0.002 s; the observation comparison grid is
  100 Hz from 0.00 s through 3.60 s inclusive.
- The equivalence tolerance is intended for the same MuJoCo version and
  deterministic source/target execution. Different MuJoCo builds, CPU
  instruction paths, or altered compiler settings should be reported with the
  actual observed tolerances rather than silently rounded to zero.
- Generated PNGs and MP4s are derived artifacts. Their provenance is recorded
  in `media/render_provenance.json` for the rendered presentation; the plotting
  script itself uses one authoritative rollout for all figures.

