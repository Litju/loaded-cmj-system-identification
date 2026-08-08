# Local release checklist

## Current status

This is a complete local Git checkout for owner review. It is not currently a
hosted repository and has no remote, DOI, or release page. That status is
intentional and is not hidden by the documentation.

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
- [x] Read-only source/target compiled-model and rollout equivalence tooling is
  available without adding the source checkout to this tree.
- [x] Tests cover model integrity, deterministic mechanics, data/preprocessing,
  and optional source equivalence.
- [x] MakeHuman source/asset acknowledgment, CC0 notice, provenance, and
  scholarly citation are included separately from the Apache-2.0 code license.

## Owner actions before external publication

These items require owner identity or hosting decisions and must not be
fabricated locally:

- [ ] Replace the generic CFF author entry with the accountable author or
  contributor list.
- [ ] Add the hosted repository URL to `CITATION.cff` once a repository exists.
- [ ] Add a DOI or archival identifier only after one has actually been issued.
- [ ] Confirm the public repository name, versioning policy, and release date.
- [ ] Reconfirm the retained MakeHuman/CC0 asset records and any additional
  third-party notices at the publication destination.
- [ ] Run the fresh-environment commands in `docs/reproducibility.md` and
  attach the final qualification record to the release review.

The absence of a URL or DOI in the current citation file is deliberate: there
is no repository to cite yet.
