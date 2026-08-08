# Mechanics

The MJCF contains a linked torso/pelvis, bilateral articulated lower limbs,
three-body feet with passive forefoot-rocker and MTP joints, hand-grip spatial
tendons, a compliant external bar/rack, six torque actuators, explicit
force-plate contact pairs, and the source sensor/site definitions.

The compiled plant contract is currently `nq=21`, `nv=21`, `nu=6`, 19 bodies,
21 joints, 28 geoms, four tendons, ten sensors, and six explicit contact pairs.
The pelvis uses bounded sagittal `root_x`, `root_z`, and `root_pitch` joints;
no external force is injected by the observation or rendering layers. Contact
forces are reconstructed from `mujoco.mj_contactForce`, rotated into world
coordinates, and aggregated with the source force-platform sign/magnitude
semantics for bilateral vertical force telemetry.
