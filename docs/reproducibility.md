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

The public experiment is `LCMJ_20KG_SYSID_V0.1.0`: every identification and
validation descriptor has `external_load_kg = 20.0`. The source-authoritative
athlete reference mass in `configs/synthetic_reference.json` is 78.37 kg. The
comparison grid is 100 Hz from 0.00 s through 3.60 s inclusive.

The generator records SHA-256 hashes in `data/dataset_manifest.json`. Generate
into a scratch directory for an audit:

```bash
LCMJ_DATASET_OUTPUT_DIR="$(mktemp -d)"
python tools/generate_dataset.py --output-dir "$LCMJ_DATASET_OUTPUT_DIR"
```

Do not use the repository itself as the output directory unless intentionally
regenerating and reviewing the tracked fixtures.

## Fresh-environment run

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python examples/simulate.py
python examples/identify.py
python examples/validate.py
LCMJ_DATASET_OUTPUT_DIR="$(mktemp -d)"
python tools/generate_dataset.py --output-dir "$LCMJ_DATASET_OUTPUT_DIR"
python examples/generate_media_suite.py
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
| Public experiment | Six fixed-20 kg identification trials and 32 fixed-20 kg validation trials |
| Bilateral measurement surface | Left/right force-platform channels are public; total force is their exact aggregate |
| Test suite | The repository test suite covers model integrity, deterministic mechanics, fixed-load fixtures, bilateral aggregation, regeneration, fitting, validation, and rendering contracts |
| Render artifact | 1280×720 MP4, 336 frames, 30 fps, 11.2 s, generated from the target plant with the MakeHuman visual family |

These are source-port and numerical reproducibility results. They are not
experimental human-validation results and do not establish physiological
validity.

## Re-run source equivalence

The source checkout is deliberately external to this repository. Set its root
at runtime and run the read-only harness:

```bash
export LOADED_CMJ_SOURCE_ROOT=/path/to/clean/source-authority
python tools/equivalence_harness.py \
  --source-root "$LOADED_CMJ_SOURCE_ROOT" \
  --trial-id 20kg_nominal_a \
  --tolerance 1e-12
LOADED_CMJ_SOURCE_ROOT="$LOADED_CMJ_SOURCE_ROOT" python -m pytest -q
```

The harness compares compiled-model structure and critical arrays, qpos/qvel,
bilateral force-platform channels, bar/LPT channels, contact states, and event
timing. It does not copy source files, write into the source checkout, or place
comparison dumps in the tracked tree. Use it only with a clean committed source
snapshot; the public release itself is qualified by the fixed-load regression
and deterministic regeneration tests.

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
- Generated PNGs and MP4s are derived artifacts. Their scenario-specific
  provenance is recorded in each `media/<scenario>/render_provenance.json`,
  with the complete media inventory and lineage summarized in
  `media/MANIFEST.json`. The suite uses one authoritative production rollout
  for all figures and the matching render.
