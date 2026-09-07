# First-submission validation

Checked September 7, 2026. All seven runs completed without retries or fallback shapes.

## Reproducibility

- A newly created Python 3.12.14 virtual environment installed the exact numerical dependency versions from the package configuration.
- Public Models 1–3 matched the original calibrated coefficients, masses, normals, predictions and final STL bytes exactly. See `calibration/clean_public_regression.json` and `calibration/clean_environment.json`.
- The standalone official Python/MATLAB evaluation reproduced all three selected public scores exactly, with mean total **1.8309487027392806**. See `calibration/official_replay.json`.
- All 24 core/extraction/certification/packaging tests passed. Three optional evaluation tests passed, including native MATLAB and Python identity controls. Ruff passed.
- The eight-page method PDF compiled without warnings; every rendered page was visually checked. See `provenance/document_qa.json`.

## Final reconstructions

All runs use lambda_W=0.03, requested K64, mass floor 1e-8, and no sparse refit.
The table reports brightness-fit diagnostics, not secret-shape accuracy.

| Model | Facets | Final brightness RMSE | Runtime (seconds) |
|---|---:|---:|---:|
| 4 | 64 | 0.08448402 | 70.6 |
| 5 | 64 | 0.05700544 | 68.8 |
| 6 | 64 | 0.13455174 | 69.4 |
| 7 | 64 | 0.11771224 | 69.8 |
| 8 | 64 | 0.04597628 | 68.6 |
| 9 | 64 | 0.06114173 | 75.1 |
| 10 | 64 | 0.18483918 | 69.2 |

Total sequential production time was 8.19 minutes on the recorded macOS arm64 CPU environment. Runtime varies by machine.

## Geometry and export checks

- All seven serialized STLs passed finite/nondegenerate geometry, exact repeated-vertex topology, watertightness, orientation, connectivity, positive volume, z contacts and cylinder contact/containment.
- All seven passed the numerical convex-boundary single-cover certificate. Its relative plane-distance and area/volume tolerances are 1e-7; sub-tolerance defects remain unresolved. It is not an exact generic self-intersection test.
- Maximum triangle-to-hull-plane distance: **1.41e-08** in challenge coordinates.
- Maximum facet log-area RMSE: **8.49e-07** (limit 1e-6).
- Maximum serialized-triangle versus transformed-measure brightness discrepancy: **2.62e-07** (limit 1e-5).
- Maximum framed closure norm: **8.09e-15** (limit 1e-10).

`results/manifest.json` binds the exact source, configuration, inputs, runtime, diagnostics and STL hashes. `results/certificates/` contains each geometry certificate. `results/SHA256SUMS` covers the package files.

## Interpretation

Convex approximation cannot recover concavities. Public examples were used during development and calibration. A bounded data-only L-curve assessment did not justify changing the shared preset; this does not prohibit a better per-model selector in a later method. No true shapes or official accuracy scores are available for Models 4–10.
