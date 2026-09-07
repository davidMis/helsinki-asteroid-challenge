# Per-model parameter assessment

Per-model hyperparameters are allowed. The first submission retains
`lambda_W=0.03` and a requested 64-facet budget after the following bounded
assessment of two input-only alternatives. The actual number of facets can
still vary when the mass-floor safeguard merges negligible cells.

The existing public calibration grid contains four regularization weights
`{0.03,0.1,0.3,1}` and four requested facet counts `{7,16,32,64}`. All other
settings remain fixed. Two rules were declared before extracting their input
diagnostics:

1. Choose lambda at an L-curve corner, retaining the 64-facet budget.
2. Choose lambda at the same corner, then choose the facet count at a second
   corner balancing complexity and approximation of the dense reconstruction.

For lambda, the coordinates are the logarithms of the dense weighted
photometric residual and the square root of its debiased Sinkhorn divergence
from the reference measure. The separate moment penalty stays at weight 0.2.
For the facet count, the coordinates are the logarithms of the actual count
and the exact chordal transport distance from dense parent to quantized child.
This second distance measures approximation of the parent, not shape accuracy.

Each rule chooses the largest positive signed Menger curvature among
consecutive interior triples, with no axis rescaling. For successive log-plane
edge vectors `a` and `b`, the curvature is
`2*det(a,b)/(|a|*|b|*|a+b|)`. Ties favor the larger parameter. Coordinates must
be positive and finite, with increasing x and nonincreasing y. Invalid curves
or curves without a positive corner fall back to the previously calibrated
setting. The complete rule and diagnostic coordinates are in `assessment.json`.

| Policy | Model 1 `(lambda,K)` | Model 2 `(lambda,K)` | Model 3 `(lambda,K)` | Public mean combined score |
|---|---|---|---|---:|
| Retained calibrated profile | `(0.03,64)` | `(0.03,64)` | `(0.03,64)` | **1.830949** |
| L-curve lambda, K=64 | `(0.1,64)` | `(0.03,64)` fallback | `(0.1,64)` | 1.821957 |
| L-curve lambda, then K | `(0.1,16)` | `(0.03,64)` fallback | `(0.1,16)` | 1.797274 |

Model 2's photometric residual is nonmonotone as lambda increases, invalidating
the two-axis rule. This need not indicate optimizer failure, because the
objective includes another fixed moment penalty. Its facet curve has no
positive corner. Models 1 and 3 have nearly power-law facet curves with very
small positive curvature at K=16; the grid provides little evidence of a
distinct complexity elbow. These weaknesses can be diagnosed without target
geometry.

The score table was joined only after input selection. It uses the certified
official public evaluation at voxel pitch 0.05 and MATLAB projection theta 0,
summing positive voxel Dice and projection score. Prior public calibration
outcomes were already known, so this is conditional calibration, not blinded
or independent validation. Final leaderboard settings remain unspecified.

These results support retaining the completed calibration for this first
submission. They do not establish that shared parameters are inherently
better. No adaptive selector is part of the production algorithm. The JSON
preserves diagnostic inputs, candidate STL hashes, and hashes of the original
development receipts; those historical files are not runtime dependencies.
