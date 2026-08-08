# Identification

`loaded_cmj.identification.identify_parameters` provides a bounded nonlinear
least-squares research example against force-platform total and bar/LPT
displacement observations, with summary and event-derived residual diagnostics.
The source reference-solution artifact supplies a deterministic parameter
configuration; it does not contain a fitting loop. This public optimizer wraps
the source plant and residual semantics without presenting that artifact as an
answer key. Parameters are addressed using the schema order; array coordinates use names such as
`joint_stiffness_Nm_rad[1]`.

The fitting result contains the complete structured parameter configuration,
the varied coordinate names, bounds-aware optimizer status, cost, residual
norm, and evaluation counts. Coordinates not selected for a particular fit stay
at the supplied initial configuration. This makes identifiability choices
explicit rather than silently claiming that every parameter is independently
recoverable from every trial.
