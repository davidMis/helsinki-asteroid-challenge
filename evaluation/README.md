# Optional official public evaluation

This adapter runs the organizers' original Python voxel evaluator and native
MATLAB `twoDmetric.m` on public Models 1–3. It is separate from reconstruction:
neither these tools nor the public target meshes are needed to produce a
submission. The downloaded official source is executed unchanged and its
September 7 snapshot checksums are verified before use.

Install the optional Python dependencies from the repository root:

```sh
python -m pip install '.[evaluation]'
```

MATLAB must have Image Processing Toolbox and Statistics and Machine Learning
Toolbox. The tested native runtime is MATLAB R2024b; the report records the
runtime and installed product versions actually used. Obtain the complete
`Evaluation_measures` directory and public target STL files from the
[official challenge data release](https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/).
They are not vendored here.

After reconstructing Models 1–3 into `runs/public`, run:

```sh
python evaluation/evaluate_public.py \
  --data-dir /absolute/path/to/released/data \
  --evaluation-dir /absolute/path/to/released/data/Evaluation_measures \
  --run-dir runs/public \
  --output-dir evaluation-results/public \
  --matlab /absolute/path/to/matlab \
  --pitch 0.05 --theta 0
```

The target layout is
`DATA/AsteroidModel01_shape_public/asteroid1.stl` and correspondingly for
Models 2 and 3. Each candidate must have a completed `modelNN/run.json` with
the matching `submission_shape.stl` checksum. `--models 1` or another subset
of `1,2,3` can be used for a smaller control run. Every invocation requires a
new output directory, so previous reports cannot silently supply results.

`evaluation.json` contains the raw voxel losses, positive scores, combined
score, input and source checksums, normalization details, and runtime
versions. The exact normalized targets, native request/result JSON, and MATLAB
log are retained alongside it. Partial progress is recorded, but a failed
run is never labeled complete. All input and implementation hashes are
checked again before completion.

## Metric and coordinate conventions

- Public target STLs retain every triangle. They are translated along z by
  the midpoint of their z extrema and divided uniformly by their z half
  extent. Their x/y origin is preserved. The normalized target is serialized
  using the same nine-significant-digit ASCII STL writer as reconstruction.
- Candidate STL bytes are unchanged; no additional translation, scale,
  decimation, or frame fitting is applied.
- The official voxel routine returns two losses. We retain both, report
  positive IoU as `1-measure1`, and use positive Dice `1-measure2` as the voxel
  score. Its bounding-box padding creates actual filled cubes; that behavior
  is preserved despite the upstream comment calling them ghost vertices.
- MATLAB independently centers each mesh by its vertex mean, rotates around
  z, and projects onto XY. Theta changes the in-plane raster orientation; it
  does not request a side-view silhouette. The original function is called
  directly for each target/candidate pair; there is no projection cache.
- The combined score is positive voxel Dice plus the projection score. Pitch
  `0.05` and theta `0` follow the released examples and our calibration. The
  organizers have not fixed final leaderboard pitch or angle aggregation in
  the published instructions used for this submission.

## Small integration controls

No challenge target is needed for the tests. They verify normalization without
x/y centering, reject a modified evaluator source, and run a translated and
scaled cube against its normalized identical reconstruction through both
original evaluators. The expected scores are exactly 1 and 1.

```sh
HAC_EVALUATION_DIR=/absolute/path/to/Evaluation_measures \
HAC_MATLAB=/absolute/path/to/matlab \
python -m unittest discover -s evaluation -p test_evaluate_public.py -v
```

Without these environment variables, the optional source/native controls are
explicitly skipped. Reconstruction and its core tests do not depend on this
evaluation directory or on MATLAB.
