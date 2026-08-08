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

## Public study identity and evidence roles

The public release is a fixed-20 kg loaded CMJ system-identification study. The
model is a synthetic rigid-body multibody abstraction. The committed source
MJCF/plant snapshot is the direct model authority; the public parameter values,
phase controller, contact settings, measurement noise, and bilateral alpha are
first-party engineering or synthetic choices recorded in the repository. No
external paper is presented as the source of those numerical values.

| Evidence item | Role | Boundary |
| --- | --- | --- |
| Committed source MJCF and direct plant port | `DIRECT_MODEL_BASIS` | Authority for morphology, contacts, actuators, integration, and telemetry semantics |
| `configs/param_schema.json` and named configurations | `PARAMETER_SOURCE` | First-party parameter contract and synthetic reference values; not a literature-derived subject model |
| `configs/preprocessing.json` and `src/loaded_cmj/measurements.py` | `MEASUREMENT_METHOD_SOURCE` | Source-defined force-platform and bar/LPT transformation and units |
| Deterministic mechanics/tests and `data/validation_trials.json` | `VALIDATION_SOURCE` | Internal computational qualification, not human validation |
| MuJoCo documentation | `SOFTWARE_OR_ASSET` | Software implementation and rendering reference |
| MakeHuman Community documentation | `SOFTWARE_OR_ASSET` | Visual asset and license pathway only |
| Briceno and Paul (2019) | `METHODS_BACKGROUND` | MakeHuman framework background only; not model-parameter provenance |

The six conditions, alpha pilot rule, seeds, same-load validation design, and
rendering outputs are synthetic engineering assumptions. They are explicitly
identified as such rather than assigned generic literature support.

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
- `media/MANIFEST.json`: scenario-level media inventory, trial identity, and
  derived-artifact hashes;
- `media/<scenario>/render_provenance.json`: renderer artifact metadata and
  sampling record for the matching production trial.

The source checkout is not vendored into the target and is not modified by the
equivalence audit.

## MakeHuman citation and acknowledgment

The athlete visual is an owner-exported MakeHuman core body asset. The original
DAE/OBJ/MTL/eye-texture files are retained, while the renderer additionally
uses generated segment and skin files documented in `MODIFICATIONS.md` and
`FILES_COPIED.md`. Those generated files are visualization transformations;
they do not enter the dynamics or measurement model.

The official MakeHuman Community license page identifies the core/exported
asset pathway as CC0, so attribution is not a legal condition for this
pathway. The official asset-pack catalog distinguishes CC0 packs from CC-BY
packs; it must not be read as a blanket license for every MakeHuman-community
asset. The inventory above contains an owner-exported core body and eye
texture, not a downloaded community asset pack. Any future addition must be
checked against its individual pack notice.

The project nevertheless cites the source because this is a research artifact.
Use both the software/asset acknowledgment and the modelling-framework
publication:

> MakeHuman Community. *MakeHuman Community*. Official project and asset
> documentation. <https://static.makehumancommunity.org/about/license.html>
> Asset-pack catalog: <https://static.makehumancommunity.org/assets/assetpacks/index.html>

> Briceno, L.; Paul, G. (2019). *MakeHuman: A Review of the Modelling
> Framework*. Advances in Intelligent Systems and Computing, 822, 224–232.
> <https://doi.org/10.1007/978-3-319-96077-7_23>

The machine-local export path records `v1py3`, but the exact MakeHuman release
was not captured in the exported metadata. This repository deliberately does
not invent a version number. The structured BibTeX records are in
[`docs/references.bib`](references.bib).
