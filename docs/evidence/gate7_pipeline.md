# Gate 7 identification and validation qualification

## Identification

The complete six-trial public identification split was passed through the
bounded 15-coordinate optimizer using the package default evaluation budget
(`max_nfev=3`, 32 residual samples per trial). The result was finite and had no
rollout failures:

| Quantity | Result |
| --- | ---: |
| trials | 6 |
| fitted coordinates | 15 |
| function evaluations | 3 |
| cost | 4.454322475887112 |
| residual norm | 2.984735323571292 |
| rollout failures | 0 |
| finite result | pass |

The optimizer reached the declared evaluation cap, so this run is an
execution/finite-residual qualification rather than a claim of a converged
scientific estimate. The fitted parameter object remains the complete
structured schema; the asymmetry alpha is not a fitted coordinate.

## Held-out validation

The synthetic reference configuration was validated against all 32 fixed-load
validation trials. All trials were valid with no exceptions or mechanics-gate
failures.

| Metric | Result |
| --- | ---: |
| trial count | 32 |
| invalid trials | 0 |
| mean total-force RMSE | 1.4410080620653534 N |
| maximum total-force RMSE | 1.7301471049310773 N |
| mean bar-displacement RMSE | 0.0001533750783856971 m |
| maximum bar-displacement RMSE | 0.0001981328566012026 m |

Nominal mechanics are independently protected by the Gate 0 fingerprints and
the exact nominal regression test in `tests/test_data_pipeline.py`.
