# Gate 0 baseline record

This record captures the unmodified target at the start of the 20 kg release
run. The source authority was inspected only at its committed snapshot; no
source-worktree changes were used.

- Start head: `b71f6fb62c52fc0cf08d6780610ed4a5b5d86371`
- Start branch: `master`
- Start worktree: clean
- Tests: 16 passed, 1 skipped (the optional source-equivalence check)
- MuJoCo: 3.11.0
- Compiled dimensions: `nq=21`, `nv=21`, `nu=6`, `nbody=19`, `njnt=21`,
  `ngeom=28`, `nsite=26`, `ntendon=4`, `nsensor=10`, `npair=6`, `neq=0`
- Timestep: `0.002 s`
- Integrator: `mjINT_IMPLICITFAST`
- Solver: 100 iterations, tolerance `1e-10`, cone `0`

The retained nominal rollout used the source-authoritative synthetic reference
configuration and `external_load_kg=20.0`. It contained 1,801 native samples
and passed finite-state, phase, flight, landing, and recovery checks.

| Quantity | Baseline |
| --- | ---: |
| movement onset | 1.568 s |
| takeoff | 2.170 s |
| landing | 2.578 s |
| takeoff velocity | 2.223280155821694 m/s |
| impulse-momentum height | 0.251935507200333 m |
| airborne duration | 0.408 s |
| propulsive impulse | 235.352260560333 N·s |
| peak concentric force | 3500.938264608662 N |
| bar peak velocity | 2.515612554405 m/s |

Canonical native-array SHA-256 fingerprints use little-endian float64 bytes
(boolean contact arrays use uint8 bytes):

| Array | SHA-256 |
| --- | --- |
| `qpos` | `84ffc29e2b9d5c99ec2e6224cc2107751455500c215489359e916585ab3db26c` |
| `qvel` | `453e6309f8088f10c34c55ac6ef31d14b430bed3f9c6b7f874067a6e651bf920` |
| `fz_left_N` | `b6e062adb2448ad5a86bcb3e2103efab1ea0885546e980e39b815cf7a0476d2e` |
| `fz_right_N` | `46f7ea5bb2c00872ab2f205ac177e84fdac916bb001aeb0f34ce71cccb7d166e` |
| `fz_total_N` | `2d406ec29eb8e27c95be2a2f6aa4b1e8ff29e6a34222b07f353e3c20bbcefc12` |
| `bar_displacement_m` | `03bf2fdfa385a7fb55e402712250ef95643782f8435864fc5bdbde2157e7e218` |
| `bar_velocity_m_s` | `c785de4142bd4eb49fde005d684335349240045ce5e3b34cd47eb94f0cdb732e` |
| left/right contact | `ff9170c5ee91272a021a873232f4d16c40401a51b6a7ac03658f104a385593fe` |

The production render also passed before publication-scope edits: 1280×720,
30 fps, 336 frames, 11.2 s, with the MakeHuman skin, bar, bilateral plate,
LPT device, force traces, phase labels, events, and metric panels.
