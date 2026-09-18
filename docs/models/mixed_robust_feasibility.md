# Mixed × Robust Relevance Pursuit feasibility

Phase 7 evaluates a categorical/continuous mixed variant of
`RobustRelevancePursuitSingleTaskGP`.

## Repository audit

The current public mixed implementations are:

- `MixedSingleTaskGP`
- `MixedSingleTaskMultiFidelityGP`
- `MixedReducedGP` and the PCA / PLS / Random Projection / AE / VAE /
  supervised AE / supervised VAE mixed reduced variants.

No mixed relevance-pursuit model is currently implemented in robotorchan.

## Decision

Do **not** expose a `MixedRobustRelevancePursuitSingleTaskGP` by composing or
renaming the existing models.

`RobustRelevancePursuitSingleTaskGP` is not merely a `SingleTaskGP` with a
different covariance module. Its robust behaviour is tied to relevance-pursuit
inference and sparse outlier/noise support. `MixedSingleTaskGP`, meanwhile,
builds a dedicated categorical/continuous covariance structure. A thin
multiple-inheritance wrapper or a mixed covariance passed to the robust model
would require verification that relevance-pursuit inference, likelihood
handling, support updates, conditioning, and BoTorch posterior semantics remain
valid.

Without that verification, a public mixed-robust class would overstate the
implemented semantics.

## Future implementation gate

A dedicated mixed robust relevance-pursuit model may be added when all of the
following are established:

1. the mixed categorical/continuous kernel is explicitly constructed and tested;
2. relevance-pursuit support selection operates correctly with that kernel;
3. raw-data retention, `make_mll()`, posterior, conditioning, and state-dict
   round trips follow robotorchan contracts;
4. acquisition functions accept the model in the original mixed input space;
5. benchmarks demonstrate robust behaviour under injected outliers for both
   continuous and categorical designs.

This is a semantic boundary, not a backward-compatibility shim. No alias,
deprecated wrapper, or nominal mixed class should be introduced meanwhile.
