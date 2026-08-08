# Provenance

This checkout is a local Git repository prepared for owner review. It has no
hosted repository, remote URL, DOI, or published release yet. The provenance
boundary is therefore expressed through tracked files, retained notices, and a
read-only external source-equivalence harness rather than through a web-hosted
release record.

The plant and renderer are direct ports of the source scientific files, with
repository paths adapted to `assets/`, `configs/`, and `src/loaded_cmj/`. The
MJCF and parameter schema remain the authority for morphology, topology,
physical arrays, names, units, bounds, and ordering.

The MakeHuman visual family is preserved under
`assets/makehuman_cmj_visual/`. Its CC0 notice, export provenance, modification
record, copied-file manifest, mesh segments, native skin scene, bone mapping,
and texture are retained. The visual model is a render-only overlay driven by
exact plant landmarks and has no effect on dynamics or measurements.

The equivalence utility in `tools/equivalence_harness.py` accepts a separately
located source checkout at runtime, so source-comparison paths and dumps are not
part of this tracked tree. Its scope is structural and behavioral equivalence:
compiled model signatures, qpos/qvel, bilateral force-platform channels,
bar/LPT channels, contact states, and event timing.

## Tracked provenance records

- `assets/loaded_cmj_model.xml`: authoritative MuJoCo model file;
- `src/loaded_cmj/plant.py`: direct scientific plant/telemetry/event port;
- `configs/param_schema.json`: parameter contract;
- `configs/preprocessing.json`: measurement and validation contract;
- `data/dataset_manifest.json`: dataset and selected artifact hashes;
- `assets/makehuman_cmj_visual/LICENSE_CC0.txt`: asset-family license notice;
- `assets/makehuman_cmj_visual/PROVENANCE.md`, `MODIFICATIONS.md`, and
  `FILES_COPIED.md`: visual asset lineage and changes;
- `media/render_provenance.json`: renderer artifact metadata and sampling
  record.

The source checkout is not vendored into the target and is not modified by the
equivalence audit.
