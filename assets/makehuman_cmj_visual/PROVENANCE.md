# MakeHuman CMJ Visual Provenance

Source export directory:
`/mnt/c/Users/Educacion/Documents/makehuman/v1py3/exports/`

Copied on: 2026-07-04T22:40:05Z

The asset was exported by the owner from MakeHuman and is included as a
render-only human skin/soft-tissue visualization for the public demonstration.

The renderer uses repository-relative paths under:
`assets/makehuman_cmj_visual/`

The MakeHuman visual model is not imported by the dynamics plant or dataset
generator. It is loaded only by `src/loaded_cmj/rendering.py` as a separate
render-only MuJoCo model, posed from live CMJ sites, and never stepped for
physics. It has no effect on plant contacts, masses, inertias, COM, CoP, GRF,
LPT signals, or trial data.
