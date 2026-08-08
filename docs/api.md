# Public API

The package exposes the source plant through a small research API. The wrappers
keep loading, simulation, preprocessing, fitting, validation, and rendering
separate while leaving the MuJoCo plant as the dynamics authority.

## Typical analysis path

```python
from loaded_cmj.dataset import load_trial
from loaded_cmj.parameters import load_named_parameters
from loaded_cmj.preprocessing import preprocess_observations
from loaded_cmj.simulation import simulate_trial
from loaded_cmj.validation import validate_trial

params = load_named_parameters("synthetic_reference")
trial = load_trial("public_001", split="identification")

observations = preprocess_observations(trial["observations"])
rollout = simulate_trial(params, trial, record=True)
report = validate_trial(params, trial)

print(rollout["traces"]["fz_total_N"])
print(rollout["events"])
print(report["trace_errors"])
```

`preprocess_observations` is the observation-side operation. `simulate_trial`
returns the native MuJoCo rollout and source measurement channels. Validation
compares those channels on the observation grid; it does not replace the
rollout with a reduced or replayed trajectory.

## Main entry points

| Entry point | Role |
| --- | --- |
| `loaded_cmj.model.build_model` | Compile the authoritative MJCF and apply a parameter configuration. |
| `loaded_cmj.model.compiled_model_signature` | Return named model counts, settings, names, and critical physical arrays for integrity checks. |
| `loaded_cmj.parameters.load_named_parameters` | Load `nominal`, `example`, `public_reference_fit`, or `synthetic_reference`. |
| `loaded_cmj.parameters.parameter_schema` | Load the canonical names, units, bounds, and vector ordering. |
| `loaded_cmj.dataset.load_trial` / `load_trials` | Load checked-in identification or validation fixtures. |
| `loaded_cmj.simulation.simulate_trial` | Run one deterministic full-morphology loaded-CMJ trial. |
| `loaded_cmj.preprocessing.preprocess_observations` | Apply the source observation transforms and weighing rule. |
| `loaded_cmj.preprocessing.comparison_grid` | Return the 100 Hz, 0.00–3.60 s inclusive comparison grid. |
| `loaded_cmj.identification.identify_parameters` | Run bounded nonlinear least squares on explicitly selected coordinates. |
| `loaded_cmj.validation.validate_trial` / `validate_parameters` | Report raw-unit trace, event, contact, and mechanics diagnostics. |
| `loaded_cmj.rendering.main` | Render the complete MakeHuman/MuJoCo scientific presentation. |

The package root re-exports the most common model, parameter, simulation, and
validation functions. The module-level imports above are preferred when a
workflow also needs dataset, preprocessing, identification, or rendering
functions.

## Identification

```python
from loaded_cmj.dataset import load_trials
from loaded_cmj.identification import identify_parameters

result = identify_parameters(
    load_trials("identification"),
    parameter_names=(
        "body_mass_kg",
        "braking_gain",
        "propulsive_gain",
        "encoder_offset_m",
    ),
    max_nfev=3,
)

assert set(result) >= {
    "parameters", "initial_parameters", "parameter_names", "success",
    "cost", "residual_norm", "nfev", "rollout_failures",
}
```

The optimizer always returns the complete structured parameter object. Only
the named coordinates are varied; all other coordinates remain at the supplied
initial configuration. The default coordinate list is defined in
`loaded_cmj.identification.DEFAULT_FIT_COORDINATES`. Bounds and coordinate
ordering come from `configs/param_schema.json`, not from the example script.

`examples/identify.py` deliberately uses two trials and two function
evaluations as a quick executable smoke example. For a research run, use all
identification trials, report the selected coordinates and optimizer budget,
and validate the resulting configuration on the separate validation split.

## Dataset generation

```python
from loaded_cmj.dataset import generate_dataset

manifest = generate_dataset(output_dir="/tmp/loaded-cmj-generated")
print(manifest["generated_file_sha256"])
```

The generator is deterministic and writes to a supplied scratch directory when
one is given. This avoids overwriting checked-in fixtures during a regeneration
audit. The generated observations use the same plant, parameter configuration,
source comparison grid, and declared noise model as the checked-in experiment.

## Rendering

The supported command-line wrapper is:

```bash
python examples/render.py
```

For direct control, the installed console script accepts the same renderer
arguments:

```bash
loaded-cmj-render \
  --output-dir media \
  --params configs/synthetic_reference.json \
  --output-file media/loaded_cmj_demo.mp4
```

Rendering is a presentation operation over a recorded plant rollout. The
MakeHuman visual asset is posed from live MuJoCo landmarks; it is not stepped
as a second physical model.

