# Measurements

The force-platform pathway reports left and right vertical force, their total,
and the net force after the weighing baseline is removed. The baseline is the
mean total vertical force over the final 1.00 s of the 1.50 s weighing interval;
system mass is `Wsys / 9.81`.

The LPT pathway reports the loaded bar's displacement at the bar LPT site,
along with its velocity, sensor delay, filtering, scale, and offset. It is a bar
measurement and is not relabeled as center-of-mass displacement. A separate LPT
tether force channel is retained for diagnostic accounting.

The source comparison grid is 0.00–3.60 s inclusive at 0.01 s spacing. Native
simulation telemetry remains available at the 0.002 s plant timestep for
mechanics and equivalence audits.

