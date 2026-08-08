# Methodology

The public experiment uses deterministic 2 ms MuJoCo rollouts of the committed
loaded-CMJ MJCF. A trial specifies external bar load, countermovement-depth and
phase-duration scales, and the source simulation duration. The plant performs a
2 s settling rollout, resets time, then records the complete trial sequence.

The observation pipeline resamples the native traces to 100 Hz, computes total
and net vertical force from the bilateral force plates, derives bar velocity by
finite differences, and preserves the source sensor delay/filter/scale/offset
rules. Bounded fitting varies named coordinates of the complete parameter vector
and re-evaluates the same full plant for every residual calculation.

Validation reports raw-unit residuals and event/mechanics diagnostics. No single
scalar is used as a scientific substitute for the force trace, bar trace,
contact sequence, or event timing.

