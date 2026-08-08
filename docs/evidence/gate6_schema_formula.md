# Gate 6 schema and formula reconciliation

The source implementation and tracked preprocessing schema agree on the
following measurement and mechanics contract.

| Quantity | Definition | Unit |
| --- | --- | --- |
| `fz_total_N` | `fz_left_N + fz_right_N` after the source measurement transform | N |
| `Wsys_N` | Mean total vertical force over the final stable 1.00 s of the 1.50 s weighing phase | N |
| `fnet_N` | `fz_total_N - Wsys_N` | N |
| `msys_kg` | `Wsys_N / 9.81` | kg |
| `bar_displacement_m` | Bar displacement at the bar/LPT site after the declared offset, scale, filtering, and delay path | m |
| `bar_velocity_m_s` | Centered finite difference of the resampled bar displacement on the 100 Hz grid | m/s |
| `propulsive_impulse_Ns` | Trapezoidal integral of `fnet_N` from the measured countermovement bottom to takeoff | N*s |
| `takeoff_velocity_m_s` | Trapezoidal integral of `fnet_N / msys_kg` from movement onset to takeoff | m/s |
| `jump_height_im_m` | `v_takeoff^2 / (2g)`, with `g=9.81 m/s^2` | m |

The reported height is therefore an impulse-momentum/takeoff-velocity-derived
jump height. It is not an optical flight-height estimate. The LPT remains a
bar-only measurement and is never interpreted as COM displacement or velocity.
The event detector uses the weighing/onset rule, sustained no-foot-contact for
takeoff, and sustained recontact for landing. Validation additionally requires
finite state, phase ordering, flight, landing, posture, drift, and contact
mechanics gates.

The comparison grid is inclusive from 0.00 s through 3.60 s at 0.01 s (100 Hz,
361 samples); the plant timestep remains 0.002 s. All public force and
kinematic channels retain their native SI units and no external literature is
claimed as the source of these repository-defined numerical thresholds or
synthetic parameter values.
