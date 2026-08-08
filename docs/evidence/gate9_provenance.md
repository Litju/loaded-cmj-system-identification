# Gate 9 provenance and citation audit

Both required evidence bundles were inspected in temporary directories only;
neither ZIP is tracked or copied into this repository.

| Bundle | SHA-256 | Checksum result |
| --- | --- | :---: |
| `loaded_cmj_scientific_provenance_evidence_bundle.zip` | `f3a4c91b211cdb002c0f1900bf949f83f3c2d53afd2396e28fa764de18e2499c` | top-level manifest valid |
| `LOADED_CMJ_SCIENTIFIC_PROVENANCE_EVIDENCE_BUNDLE_20260808.zip` | `9701c0a70b19216204daf50fa41a2cb6e49899c6df43affe1510bbab8906e535` | top-level manifest valid |

The newer 20260808 reconstruction receives greater evidentiary weight. Its
top-level `SHA256SUMS.txt` validated all 64 included payloads. Three nested
protocol manifests also validated. The nested RCCP-A manifest references
checkpoint/diagnostic/raw payloads intentionally absent from the archive, so it
is not a self-contained manifest; this is recorded as a partial nested-manifest
limitation, not a top-level bundle failure.

## Reconciliation against primary evidence

Primary project evidence and the live target Git history override stale bundle
metadata. The target execution began at
`891d987021750f6b6d2a97c0e55b0f2be0548e0c` on
`release/20kg-sysid-v0.1.0`; the source authority was inspected at
`470d1666ce37407b8898c374170921e5d47cc5ff`. The bundle's older recorded target
HEAD and its historical reduced-dimension descriptions were not used to make
physical-model decisions. The current target authority is the compiled
`nq=21`, `nv=21`, `nu=6` plant, 27-scalar parameter schema, 78.37 kg synthetic
reference body mass, and six/32 fixed-load datasets.

The newer bundle also asks for a local confirmation of SciPy use. The target
code directly imports `scipy.optimize.least_squares`, so SciPy is classified as
an actually used software dependency here. No numerical plant parameter was
copied from SciPy, MuJoCo, or any biomechanics source.

## Reference-role classification

This table separates the role assigned by the evidence audit from the stronger
claim that a source supplied a model parameter. The target repository has no
external `DIRECT_MODEL_BASIS` or `PARAMETER_SOURCE`; morphology, numerical
values, actuator semantics, contact settings, sensor imperfections, bounds,
seeds, event thresholds, and the selected alpha are first-party synthetic
engineering decisions.

| Reference group | Role | Target-release disposition |
| --- | --- | --- |
| Target MJCF, direct plant port, and primary source history | `DIRECT_MODEL_BASIS` | Actual model authority; not a literature citation |
| Target parameter schema/configurations and generator | `PARAMETER_SOURCE` | First-party synthetic parameter source; no external numerical source claimed |
| Hamill, Knutzen & Derrick; Zatsiorsky, *Kinetics of Human Motion* | `MEASUREMENT_METHOD_SOURCE` in the newer bundle | Methodological interpretation in the audit bundle; not used to set target numerical values and not added to the target bibliography without a target-source binding |
| Komi, *Strength and Power in Sport* | `VALIDATION_SOURCE` in the newer bundle | Validation background in the audit bundle; not a source of plant parameters |
| Sutton & Barto; Nocedal & Wright; Lynch & Park; Tedrake; Featherstone; Shabana; Zatsiorsky, *Kinematics*; Khalil; Åström & Murray; Särkkä & Svensson; Ljung; Beck; Brogliato | `METHODS_BACKGROUND` in the newer bundle | Evidence for later verification doctrine; not a numerical model basis |
| Todorov, Erez & Tassa; NumPy; SciPy; MuJoCo documentation | `SOFTWARE_ASSET_REFERENCE` | Actually used software/documentation; cited in `docs/references.bib` |
| MakeHuman Community documentation | `SOFTWARE_ASSET_REFERENCE` | Render-only reused asset and license provenance; no physics effect |
| Briceno & Paul (2019) | `GENERAL_BACKGROUND` | MakeHuman framework background only; no model-parameter provenance |
| References present only as unbound candidates in the older bundle or source library | `NOT_ACTUALLY_USED` | Not added to the target bibliography |

The target bibliography therefore contains only sources with an evidenced
release role: MuJoCo, NumPy, SciPy, MakeHuman documentation, and the MakeHuman
framework paper. The first-party assumption boundary is retained explicitly in
`docs/provenance.md` and the asset license/provenance files.

## Asset and repository licensing

The repository code and documentation are Apache-2.0 under `LICENSE` and the
current `CITATION.cff`. The MakeHuman visual family is retained separately with
its CC0 notice, copied-file inventory, modification record, and render-only
boundary. The owner-exported asset's exact application release was not captured;
the local `v1py3` path is not promoted to a version claim.
