# Gate 9 provenance and citation audit

Both owner-supplied evidence bundles were inspected in temporary directories
only. Neither ZIP, raw literature PDF, or private evidence report is tracked or
copied into the public repository.

| Bundle | SHA-256 | Inventory | Checksum result |
| --- | --- | ---: | :---: |
| `loaded_cmj_scientific_provenance_evidence_bundle.zip` | `f3a4c91b211cdb002c0f1900bf949f83f3c2d53afd2396e28fa764de18e2499c` | 17 files | top-level manifest valid |
| `LOADED_CMJ_SCIENTIFIC_PROVENANCE_EVIDENCE_BUNDLE_20260808.zip` | `9701c0a70b19216204daf50fa41a2cb6e49899c6df43affe1510bbab8906e535` | 64 files | top-level manifest valid |

The newer archive's `SHA256SUMS.txt` validated its 63 payload checksum lines;
the 64th archive member is the checksum file itself. Its RCCP-A protocol
sidecar and the RD1 checkpoint/protocol sidecars also validate. The nested
`RCCP_A_SHA256SUMS.txt` ledger is not self-contained: 33 listed checkpoint,
diagnostic, and raw payloads are intentionally absent from the archive. That
partial nested-ledger limitation is recorded rather than silently treated as a
successful full archive validation.

## Reconciliation against primary evidence

The frozen target is the current committed release at
`2e469217777f4150b834c03ab5f4a46f9892366b`, with the 21-DOF compiled plant
contract (`nq=21`, `nv=21`, `nu=6`) and the 27-scalar schema protected by the
tracked integrity tests. A stale 24-parameter statement in the newer
reconstruction was corrected from the committed schema. The current target
directly imports `scipy.optimize.least_squares`, resolving the newer bundle's
request for local confirmation of SciPy use.

The older bundle's conservative bibliography retained only MuJoCo, NumPy, and
two split MakeHuman entries. The newer bundle contains explicit source-binding
records for the methods, measurement, and validation references. Primary
committed evidence supports retaining those references as methods/measurement/
validation provenance, while still rejecting them as parameter sources. The
two MakeHuman entries are consolidated into one software/asset reference.

The newer bundle's Featherstone DOI was corrected from
`10.1007/978-0-387-74315-8` (the publisher record for a different book) to
`10.1007/978-1-4899-7560-7`. The newer ledger's software entries were
reclassified as `SOFTWARE_ASSET_REFERENCE`; its MakeHuman entry was separated
from the general-background MakeHuman review. The RL/task-authoring-only
reference was excluded from the public bibliography. Matplotlib was retained
from the primary target because the checked-in publication tooling imports it.

## Final role boundary

The committed MJCF/plant/schema are first-party `DIRECT_MODEL_BASIS` and
first-party parameter authority. No external literature source is retained as
`DIRECT_MODEL_BASIS` or `PARAMETER_SOURCE`. The final public bibliography is
the citation-only set documented in [`docs/reference_map.md`](../reference_map.md):

- measurement methods: Hamill/Knutzen/Derrick and Zatsiorsky, *Kinetics*;
- validation: Komi, *Strength and Power in Sport*;
- methods background: rigid-body/contact, numerical, stability, sampled-data,
  and system-identification references;
- software/assets: MuJoCo, NumPy, SciPy, Matplotlib, and MakeHuman;
- general background: Briceno and Paul's MakeHuman review.

No source supplies the target's morphology, masses, inertias, contact or
actuator constants, sensor calibration/noise, event thresholds, parameter
bounds, fixed 20 kg condition, or `alpha=0.02`. Those choices are explicitly
classified in the public provenance and reference-map documents. `alpha=0.02`
is the smallest mechanically valid tested synthetic excitation, not a
physiological or clinical threshold.

## Copyright and license boundary

Scientific literature remains citation-only. The repository contains no
third-party paper or book PDFs, copied figures, copied tables, substantial
quotes, or copied copyrighted pseudocode. The MakeHuman visual asset remains a
separate tracked asset/license chain under `assets/makehuman_cmj_visual/`.
