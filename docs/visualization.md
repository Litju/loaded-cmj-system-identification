# Rendering and visualization

The project has two related but distinct presentation paths:

1. `src/loaded_cmj/rendering.py` renders the animated MuJoCo scene and its
   scientific dashboard into an MP4.
2. `examples/plot_observables.py` renders static diagnostics from one
   authoritative rollout and the public observations.

Both paths use the same plant, parameter configuration, measurement names,
events, and phase timing. The plotting layer does not run a second model or
invent a plotting-only trajectory. It reads existing trace, summary, event,
and observation fields; display markers and translucent phase bands are visual
annotations only.

## Source-defined CMJ phase dissection

The hard-coded phase timing returned by the plant is shown behind every time
plot and in the phase/event figure:

| Phase | Interval (s) | Interpretation |
| --- | ---: | --- |
| Weighing | 0.00–1.50 | Quiet baseline and initial support measurement |
| Unweighting | 1.50–1.82 | Movement begins and force is reduced |
| Braking | 1.82–2.12 | Downward motion is arrested |
| Propulsion | 2.12–2.58 | Supported force production toward takeoff |
| Flight | 2.58–2.95 | Bilateral no-contact interval where physically detected |
| Landing absorption | 2.95–3.25 | Recontact and impact absorption |
| Stabilization | 3.25–3.60 | Post-landing recovery window |

The event lines are measured from the rollout. A scheduled phase boundary is
not substituted for a missing physical takeoff or landing event.

## Static figure set

Run:

```bash
python examples/plot_observables.py
```

The script writes:

| File | Contents |
| --- | --- |
| `media/observable_fit.png` | Force-platform, bar/LPT, tether, and observed/predicted comparison panels |
| `media/combined_grf_com_lpt.png` | One physical plotting rectangle combining bilateral/total GRFs, COM-z, LPT velocity, and LPT displacement with native-unit twin y-axes |
| `media/force_plate_metrics.png` | Bilateral, regional, and horizontal contact-force channels |
| `media/global_kinematics.png` | Root, COM, bar, and global velocity channels |
| `media/foot_kinematics.png` | Foot landmarks and bilateral clearance |
| `media/joint_kinematics.png` | Anatomical, foot, and bar-rack positions and velocities |
| `media/contact_mechanics.png` | Contact state, center of pressure, and slip diagnostics |
| `media/phase_events.png` | Phase index and measured event timing |
| `media/summary_metrics.png` | Existing scalar summaries and diagnostics |

The combined figure uses multiple y-axes because force is measured in newtons
while COM/LPT channels are measured in metres or metres per second. It does not
normalize those channels into a dimensionless score. The other figures keep
force-platform mechanics, bar/LPT measurements, and kinematics separated by
scientific role, with joint and bar-rack coordinates grouped as kinematics.

## Visual encoding

- black/dark foreground: plotting surface and panels;
- observed signal: light blue family;
- MuJoCo prediction/state: marker-bearing traces with channel-specific colors;
  yellow is the primary MuJoCo key where it does not collide with another
  channel;
- force channels, COM, LPT velocity, and LPT displacement: distinct colors
  within each axes so channels remain distinguishable;
- translucent colored background: source-defined CMJ phase;
- light dashed vertical markers: movement onset, takeoff, and landing when
  physically detected.

The color choices are a visual encoding, not a change to the measurements.
Native units and source channel names remain in the axis labels and legends.

## Animated presentation

The renderer preserves the approved MakeHuman visual model and overlays the
live external bar, bilateral force plates, LPT device/tether, COM marker/path,
phase timeline, force traces, bar/LPT traces, events, joint information, and
metric panels. It writes 1280×720, 16:9, 30 fps H.264 MP4 output by default.

The visual model is render-only. The physical trajectory comes from the
compiled MuJoCo plant; the skin/segment assets do not alter masses, contacts,
actuators, sensors, or measurements.
