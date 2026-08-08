# Public release checklist

## Current status

Loaded CMJ System Identification is publicly hosted at:

https://github.com/Litju/loaded-cmj-system-identification

The repository identifies the software as version `0.1.0`, includes the
accountable author in `CITATION.cff`, and publishes the source repository URL.
No DOI or independent archival identifier is currently assigned.

## Scientific and engineering checks

- [x] Full source-authoritative MuJoCo plant is tracked under `assets/` and
  `src/loaded_cmj/`.
- [x] MakeHuman mesh, skin, license, provenance, modification, and copied-file
  records are tracked under `assets/makehuman_cmj_visual/`.
- [x] Parameter schema, nominal/example/reference configurations, preprocessing
  contract, and explicit identification/validation fixtures are tracked.
- [x] Dataset generation is deterministic and supports scratch output.
- [x] Bounded identification and mechanics-first validation are exposed through
  normal research modules.
- [x] Static figures and the animated renderer use the same target plant.
- [x] Tests cover model integrity, deterministic mechanics, data/preprocessing,
  validation, and publication-visualization contracts.
- [x] MakeHuman source/asset acknowledgment, CC0 notice, provenance, and
  scholarly citation are included separately from the Apache-2.0 code license.
- [x] The publication visualization pass preserves model-versus-observed and
  left-versus-right trace identity using redundant semantic encoding.
- [x] Numerical, event, and metric regression checks passed after the final
  visualization refactor.
- [x] Final visualization-refactor test suite: 29 passed, 1 optional
  source-equivalence test skipped.
- [x] Final worktree qualification reported no plant, physics, dataset,
  scenario, identification, validation-logic, or metric-calculation changes.

## Public repository identity

- [x] Public repository created:
  `Litju/loaded-cmj-system-identification`.
- [x] Accountable author recorded as Julio Rodriguez.
- [x] Repository URL recorded in `CITATION.cff`.
- [x] Software version recorded as `0.1.0`.
- [x] Apache-2.0 repository license retained.
- [x] MakeHuman asset provenance and separate license notice retained.
- [x] Public documentation no longer represents the project as an unhosted
  local release candidate.

## Remaining release and archival actions

- [ ] Create or verify the annotated `v0.1.0` Git tag.
- [ ] Create or verify the corresponding GitHub Release.
- [ ] Add a release date to citation metadata when the release identity is finalized.
- [ ] Add a DOI or archival identifier only if and when one is actually issued.
- [ ] Recheck GitHub-rendered README links, media, citation metadata, and release
  assets after the final documentation commit.

A DOI is not required for the GitHub source release. If an archival identifier
is assigned later, update `CITATION.cff` and the release documentation in a
new, traceable commit.
