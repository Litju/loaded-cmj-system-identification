# Rendering and visualization

The repository has two presentation paths:

1. `src/loaded_cmj/rendering.py` renders the approved MuJoCo/MakeHuman animated
   scene and dashboard.
2. `examples/plot_observables.py` renders the publication static diagnostics
   from one authoritative rollout and its public observations.

`examples/generate_media_suite.py` applies the static plotting path to the six
qualified identification scenarios. It builds common display limits from the
complete six-scenario set, then writes one self-contained bundle per scenario.
The plotting layer does not change measurements, resample signals, run a
second model, or normalize away native units. `media/MANIFEST.json` records the
plot hashes, source-data hashes, scale policy, and existing render lineage.

## Semantic visual grammar

Color identifies a physical channel or anatomical region. Important identity is
also encoded geometrically:

| Meaning | Encoding |
| --- | --- |
| MuJoCo/model | continuous clean line |
| Observed | same physical-channel hue with sparse open-circle markers |
| Left/right anatomy | left solid; right dashed |
| Foot region | heel blue; forefoot green; toe/MTP orange |
| Joint family | hip blue; knee vermillion; ankle green; rocker purple; MTP orange |
| Bilateral force comparison | left sky blue; right orange; total off-white |
| Aggregate COM | warm gold |

The palette is fixed in `examples/plot_observables.py`. It is a dark,
color-blind-safe starting palette with line style, markers, panel position, and
legend semantics providing redundant identity. It is a visual encoding only;
it does not alter the scientific channels.

Phase bands are quiet top ribbons on dense signal plots. `phase_events.png`
retains stronger phase fills because phase structure is its subject. Event lines
use a consistent dashed convention for movement onset, takeoff, and landing.

## Publication static figure family

Each qualified scenario directory contains:

| File | Contents |
| --- | --- |
| `observable_fit.png` | Bilateral plate, aggregate GRF, bar/LPT displacement and velocity, and tether panels with model/observed source encoding |
| `combined_grf_com_lpt.png` | Four synchronized native-unit panels: GRF, COM vertical position, bar/LPT displacement, and bar/LPT velocity |
| `force_plate_metrics.png` | Bilateral plate forces, regional vertical force, regional horizontal force, and total horizontal force |
| `global_kinematics.png` | Root, COM, bar, and global velocity channels |
| `foot_kinematics.png` | Separate left/right landmark panels and minimum bilateral clearance |
| `joint_kinematics.png` | Five-row lower-limb matrix with position and velocity columns; each axis contains only left/right traces |
| `auxiliary_kinematics.png` | Trunk, shoulder/elbow, bar-rack translation, and bar-rack pitch panels separated by dimensional family |
| `contact_mechanics.png` | Six-lane contact raster, centre of pressure, and two-trace slip diagnostics |
| `phase_events.png` | Source phase index and measured event timing |
| `summary_metrics.png` | Performance/events, mechanical/numerical integrity, and bilateral-condition blocks |

Only `20kg_bilateral_asymmetry` additionally contains
`bilateral_force_asymmetry.png`, which preserves the fixed-load bilateral
comparison and direct event-window impulses in N·s. For non-asymmetry scenarios,
asymmetry-specific summary rows say `not applied`; they are not represented as
misleading zeros.

`LPT displacement` means bar displacement and `LPT velocity` means bar
velocity. LPT is not a COM measurement. The combined figure uses aligned
panels instead of multiple y-axes, and no axis mixes metres with radians or
their rates.

## Phase and event source

The source-defined phase intervals are shown as context:

| Phase | Interval (s) |
| --- | ---: |
| Weighing | 0.00–1.50 |
| Unweighting | 1.50–1.82 |
| Braking | 1.82–2.12 |
| Propulsion | 2.12–2.58 |
| Flight | 2.58–2.95 |
| Landing absorption | 2.95–3.25 |
| Stabilization | 3.25–3.60 |

Event lines are taken from the rollout event dictionary. A scheduled phase
boundary is not substituted for a missing physical takeoff or landing event.

## Regeneration and provenance

To regenerate the static plots while reusing the already-qualified videos:

```bash
PYTHONPATH=src:examples uv run --isolated --no-project --python 3.13 \
  --with 'matplotlib==3.11.1' --with 'numpy==2.5.1' \
  --with 'scipy==1.18.0' --with 'mujoco>=3.2,<4' \
  python examples/generate_media_suite.py --reuse-renders
```

The six scenario identifiers are fixed by `data/dataset_manifest.json` and
must remain:

`20kg_nominal_a`, `20kg_nominal_b`, `20kg_depth`, `20kg_timing`,
`20kg_depth_timing`, and `20kg_bilateral_asymmetry`.

The animated renderer remains unchanged by the static visualization refactor.
Its telemetry, scalar summary, and render provenance use the same positive
upward takeoff-velocity convention. No rendering video is regenerated when the
renderer is untouched.

The figures are synthetic/model-based presentation outputs. They do not imply
experimental human validation, clinical validation, or measurement uncertainty
that is not present in the source observations. All force, displacement,
velocity, impulse, and angular quantities retain their native units in labels.
