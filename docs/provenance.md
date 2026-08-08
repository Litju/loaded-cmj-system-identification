# Provenance

The public plant and renderer are direct ports of the source scientific files,
with repository paths adapted to `assets/`, `configs/`, and `src/loaded_cmj/`.
The MJCF and parameter schema remain the authority for morphology, topology,
physical arrays, names, units, bounds, and ordering.

The MakeHuman visual family is preserved under
`assets/makehuman_cmj_visual/`. Its CC0 notice, export provenance, modification
record, copied-file manifest, mesh segments, native skin scene, bone mapping,
and texture are retained. The visual model is a render-only overlay driven by
exact plant landmarks and has no effect on dynamics or measurements.

The equivalence utility in `tools/equivalence_harness.py` accepts a separately
located source checkout at runtime, so source-comparison paths and dumps are not
part of the public tracked tree.

