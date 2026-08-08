# Gate 5 dataset regeneration

The checked-in public outputs were regenerated with
`tools/generate_dataset.py` into a temporary scratch directory and compared
byte-for-byte with the repository files. The generator is deterministic
(`generation_seed=503117`, `generator_version=1.5.0`) and uses the same 2 ms
MuJoCo plant as the checked-in fixtures.

| Output | SHA-256 | Scratch comparison |
| --- | --- | :---: |
| `data/identification_trials.json` | `1eace786854fb958bbf07611a7701c09fb4218c8e270a96e1fd2c479ef7cdcb3` | pass |
| `data/validation_trials.json` | `e7287d99cfa3e44bf17ca8e6d4508817f405ab51ae697228cdd40ebfcf26d015` | pass |
| `data/dataset_manifest.json` | `d4c2d3cf7ec674f263d755aff1157c922b1c45a989a2060428c477ede93d9d73` | pass |
| `configs/preprocessing.json` | `f69c90cc30fbf8055619b2700db797d8746259bbbd7490ca79011820f1b47856` | pass |
| `configs/synthetic_reference.json` | `8b8eb825ad8c1960dcac1eca18c0f9a1116dc46fcbadfaf2fb88928064d7d5a6` | pass |

The public split contains six identification trials and the validation split
contains 32 trials. Every descriptor has `external_load_kg=20.0` and
`bar_load_kg=20.0`; the only nonzero known excitation is the frozen
`synthetic_interlimb_drive_asymmetry` trial at alpha `0.02`.
