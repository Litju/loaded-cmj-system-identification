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
