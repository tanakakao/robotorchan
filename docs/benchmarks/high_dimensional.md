# High-dimensional benchmark problems

These deterministic, tensor-native minimization problems provide
controlled ambient dimension and effective dimensionality.

| Problem | Ambient dimension | Active structure | Known minimum |
| --- | ---: | --- | ---: |
| sparse_sphere_50 | 50 | first four coordinates; 46 irrelevant | 0 |
| rotated_subspace_40 | 40 | two dense orthogonal projections | 0 |
| interaction_chain_30 | 30 | ten coordinates with adjacent pair interactions | 0 |

**Sparse sphere:** sum of squared deviations from 0.3 in the first
four coordinates. The remaining 46 dimensions have no effect.

**Rotated subspace:** the squared norm of two projections of
`X - 0.5` onto normalized all-ones and half-positive/half-negative
directions. The optimum is achieved at the center; the objective is
constant along the 38-dimensional null space.

**Interaction chain:** squared deviations from 0.4 in the first ten
coordinates, plus squared differences between adjacent deviations.
The remaining 20 dimensions are irrelevant.

All problems use bounds [0, 1] in every dimension. Known minima allow
truth-based simple regret comparisons between standard GP, SAAS,
projection-based and trust-region search strategies. This phase adds
the problem definitions and integration tests, not a benchmark ranking.

Register using `register_high_dimensional_problems(registry)`.
