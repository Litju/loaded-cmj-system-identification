# Mechanics

The MJCF contains a linked torso/pelvis, bilateral articulated lower limbs,
three-body feet with passive forefoot-rocker and MTP joints, hand-grip spatial
tendons, a compliant external bar/rack, six torque actuators, explicit
force-plate contact pairs, and the source sensor/site definitions. This is the
full source morphology and contact topology; the public API does not introduce
a telescoping-leg, capsule-only, or toy substitute.

The compiled plant contract is currently `nq=21`, `nv=21`, `nu=6`, 19 bodies,
21 joints, 28 geoms, four tendons, ten sensors, and six explicit contact pairs.
The pelvis uses bounded sagittal `root_x`, `root_z`, and `root_pitch` joints;
no external force is injected by the observation or rendering layers. Contact
forces are reconstructed from `mujoco.mj_contactForce`, rotated into world
coordinates, and aggregated with the source force-platform sign/magnitude
semantics for bilateral vertical force telemetry.

Rigid-body, wrench, and contact-frame interpretation is literature-informed by
`@featherstone_2008`, `@shabana_2010`, and `@lynch_park_2017`; the compiled MJCF
and all numerical values remain first-party model authority.

The public experiment uses a fixed `external_load_kg = 20.0`. The public
synthetic reference configuration uses `body_mass_kg = 78.37 kg`; the 75 kg
value retained in the plant source is the committed morphology's mass-scaling
reference, not a second public experiment condition. No plant morphology,
contact geometry, mass/inertia semantics, timestep, or solver setting is
changed by the publication-scope freeze.

## State and parameter boundary

The MuJoCo state is authoritative for positions, velocities, contact, bar
position, and COM. `model.build_model` compiles `assets/loaded_cmj_model.xml`
and the plant applies the structured configuration from `configs/`. The
parameter names do not stand for independent biological muscle properties; they
are parameters of this rigid-body/contact/actuator and measurement model.

The control path uses the source six torque actuators with phase-aware position
tracking, feed-forward, and trunk-posture stabilization. Passive forefoot/MTP
articulation remains passive. Observation collection is observationally pure:
it reads MuJoCo state and contact wrenches without writing `qpos`, `qvel`, or
external force arrays after initialization.

The qualified bilateral condition is applied after this symmetric six-actuator
controller in the trial layer: left commands are scaled by `1 + alpha` and
right commands by `1 - alpha`, with compiled actuator bounds enforced. Alpha is
a known synthetic excitation and is absent from the fitted parameter vector.

## Contact and event semantics

Foot-region contact is derived from the eligible MuJoCo contact geometries and
world-frame contact wrenches. The public event path uses force/contact telemetry:
movement onset follows the weighing threshold, takeoff requires sustained
bilateral no-contact after propulsion, and landing follows sustained
recontact. A scheduled phase boundary is not treated as a physical event when
the contact guard is absent.

The complete compiled-model and rollout comparison is documented in
[`docs/reproducibility.md`](reproducibility.md) and implemented by
`tools/equivalence_harness.py`.
