# Gate 3 bilateral asymmetry qualification

The asymmetry condition is a known trial-layer excitation named
`synthetic_interlimb_drive_asymmetry`. It scales the compiled six-actuator
command after the existing controller computes it: left-leg commands use
`1 + alpha` and right-leg commands use `1 - alpha`, subject to the existing
MuJoCo actuator ranges. The articulated plant, contacts, solver, integrator,
measurement model, and event rules are unchanged.

The pilot panel was restricted to the required values `alpha = 0.02`, `0.04`,
and `0.06`, with a fixed 20.0 kg external load and the synthetic reference
parameters. Qualification required finite deterministic replay, valid
countermovement/propulsion/takeoff/flight/landing/recovery mechanics, complete
phase observation, bilateral support contact, a sustained no-foot-contact
flight interval, zero actuator clipping, and measurable separation in the
source-authoritative L/R force traces. The release audit uses an event-window
propulsive impulse separation greater than 5 N*s as the measurable-separation
criterion. This is a computational qualification threshold, not a clinical
cutoff or an inferential statistical test.

| alpha | takeoff (s) | landing (s) | depth (m) | flight (s) | L propulsive impulse (N*s) | R propulsive impulse (N*s) | L-R (N*s) | clip count | mechanics | exact replay |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 0.02 | 2.172 | 2.580 | 0.202700 | 0.408 | 120.395827 | 113.534855 | 6.860972 | 0 | pass | pass |
| 0.04 | 2.172 | 2.580 | 0.202697 | 0.408 | 123.661219 | 110.331833 | 13.329387 | 0 | pass | pass |
| 0.06 | 2.172 | 2.580 | 0.202690 | 0.408 | 126.896715 | 107.170585 | 19.726130 | 0 | pass | pass |

All three pilots had finite states, all seven phase indices, both left/right
contact channels, no-contact flight, bounded torso motion, posterior drift
below the mechanics threshold, and no failed mechanics gates. The smallest
qualifying value is therefore frozen as `alpha = 0.02`. The public asymmetry
trial exposes direct `fz_left_N`, `fz_right_N`, `fz_total_N`, and side-specific
braking/propulsive impulses; alpha is not a fitted identification coordinate.

Because this panel contains one deterministic rollout condition per alpha, the
result supports mechanics qualification and reproducibility only. It does not
support population inference, clinical interpretation, or an uncertainty
interval for asymmetry.
