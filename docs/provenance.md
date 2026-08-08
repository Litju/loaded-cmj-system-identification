# Provenance

This release is a fixed-20-kg, synthetic Loaded CMJ system-identification
study. The public model is a first-party rigid-body/contact/actuator and
measurement abstraction. The citation boundary is intentionally explicit:
published sources inform methods, measurement interpretation, validation
language, and software/asset acknowledgment; they do not supply the plant's
numerical parameters.

The complete source-to-code records are in
[`docs/reference_map.md`](reference_map.md). This document states the public
scientific boundary without exposing development-task archaeology.

## FIRST-PARTY IMPLEMENTED MODEL

The scientific authority is the committed repository implementation:

- `assets/loaded_cmj_model.xml` defines the linked sagittal topology, bilateral
  lower limbs, articulated hindfoot/forefoot/toe structure, contacts, bar/rack,
  tendons, actuators, sensors, timestep, and solver settings.
- `src/loaded_cmj/plant.py` and `src/loaded_cmj/clean_core.py` implement forward
  dynamics, phase-aware control, contact-wrench collection, force-platform
  channels, bar/LPT channels, events, impulse, and summary mechanics.
- `src/loaded_cmj/model.py` and `tests/test_model_integrity.py` protect the
  compiled model contract (`nq=21`, `nv=21`, `nu=6` and the recorded topology
  and solver settings).
- `configs/param_schema.json`, the named configurations, and
  `tools/generate_dataset.py` are the first-party parameter and synthetic-data
  authorities. The current identification schema has 27 scalar coordinates.

These files are `DIRECT_MODEL_BASIS` and, where applicable, first-party
`PARAMETER_SOURCE` evidence. They are not entries in the literature
bibliography. No external paper is claimed as the source of the morphology,
segment geometry, COMs, inertias, joint ranges, actuator values, contact
values, bar/rack values, sensor imperfections, event thresholds, or fitted
bounds.

## LITERATURE-DERIVED ELEMENTS

No retained external reference is classified `DIRECT_MODEL_BASIS` or
`PARAMETER_SOURCE`. Consequently, this release makes no claim that the model
is “based on” a named anthropometry, contact, actuator, or biomechanics source.
The model's topology and numerical values remain first-party engineering
choices and synthetic data-design choices.

## LITERATURE-INFORMED METHODS

The methods bibliography records conceptual guidance that was explicitly used
during development and verification:

- rigid-body, constrained, floating-base, and contact reasoning:
  `@lynch_park_2017`, `@tedrake_underactuated_2024`, `@featherstone_2008`,
  `@shabana_2010`, and `@brogliato_1996`;
- sampled measurement and plant/controller/sensor separation:
  `@zatsiorsky_kinematics_1998`, `@astrom_murray_2008`, and
  `@sarkka_svensson_2023`;
- bounded optimization, model structure, system identification, and
  validation discipline: `@nocedal_wright_2006`, `@ljung_1999`, and
  `@beck_1979`.

These references support method vocabulary and verification reasoning. They do
not imply reproduction of a named algorithm beyond what the committed code
actually does, and they do not provide task-specific numerical values.

## LITERATURE-INFORMED VALIDATION

`@hamill_knutzen_derrick_2015` and `@zatsiorsky_kinetics_2002` informed the
interpretation of force-platform signals, external/internal force, bilateral
force, impulse, and COM-related mechanics. `@komi_2003` informed
literature-aware plausibility checks for CMJ/SSC/landing mechanics.

The reported validation is still internal and synthetic: deterministic
mechanics validation, numerical validation, replay/determinism validation, and
held-out same-plant validation. It is not experimental human validation,
clinical validation, population inference, or subject-specific validation.

## NUMERICAL/ENGINEERING DESIGN CHOICES

The release deliberately exposes, rather than disguises, its unreferenced
choices. They include:

- morphology, topology, geometry, COM/inertia layout, mass scaling, and joint
  ranges;
- the loaded-CMJ phase schedule, countermovement depth, braking/propulsion
  sequencing, takeoff/landing/recovery rules, and fixed 20 kg condition;
- six torque actuators, PD/feed-forward control, stiffness/damping, activation
  delay and time constants;
- foot-contact geometry, friction, soft-contact settings, thresholds, and
  force aggregation;
- compliant rack, bar/hand coupling, attachment geometry, and bar displacement;
- force scale/bias, synthetic noise, 100 Hz comparison grid, filtering, delay,
  finite differences, offsets, and time alignment;
- the 27-scalar identification structure, bounds, residual composition, and
  six-identification/32-validation synthetic split;
- the 0.002 s timestep, solver/integrator settings, settling procedure, seeds,
  and synthetic distributions.

The bilateral `alpha=0.02` is a known trial-layer excitation. It was selected as
the smallest mechanically valid value in the tested synthetic panel and is a
`VALIDATION_CONSTRAINED_CHOICE`/`SYNTHETIC_DATA_DESIGN`, not a normative,
clinical, pathological, dominant-limb, injury-risk, or physiological
threshold. The fixed 20 kg load is the experiment condition chosen by this
project, not a literature-derived recommended load.

The full 51-item inventory and classifications are maintained in
[`docs/reference_map.md`](reference_map.md). All public quantities retain SI
units. Synthetic noise defines deterministic realizations; it is not an
empirical instrument-uncertainty estimate, confidence interval, or population
error bar.

## SOFTWARE DEPENDENCIES

The materially used software references are:

- `@todorov_erez_tassa_2012` for MuJoCo, including the actual `mj_step` and
  `mj_contactForce` dependency;
- `@harris_numpy_2020` for NumPy array operations;
- `@virtanen_scipy_2020` for the direct `scipy.optimize.least_squares` use in
  `src/loaded_cmj/identification.py`;
- `@hunter_matplotlib_2007` for the checked-in scientific plot and media
  tooling.

The official MuJoCo documentation is a runtime documentation URL, not a second
model-provenance source. Transitive packages and test utilities are not given
separate scientific citations.

## ASSET PROVENANCE

The MakeHuman visual family is a separate, render-only asset chain. Its tracked
records are:

- `assets/makehuman_cmj_visual/PROVENANCE.md`;
- `assets/makehuman_cmj_visual/LICENSE_CC0.txt`;
- `assets/makehuman_cmj_visual/MODIFICATIONS.md`; and
- `assets/makehuman_cmj_visual/FILES_COPIED.md`.

The renderer loads and poses this visual family from live plant landmarks; it
does not step it and it has no effect on plant contacts, masses, inertias, COM,
CoP, GRF, LPT signals, data, or scoring. `@makehuman_community_2026` is the
software/asset citation. `@briceno_paul_2019` is general MakeHuman framework
background and asset acknowledgment only. The exact local MakeHuman
application release was not recorded, so no version is invented.

The repository ships the asset notices and citation metadata only. Scientific
literature is citation-only: no article or book PDFs, copied figures, copied
tables, substantial quotations, or copied copyrighted pseudocode are included.
