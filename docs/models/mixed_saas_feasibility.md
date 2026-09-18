# Mixed × SAAS feasibility

Phase 6 evaluates whether robotorchan should expose a mixed categorical/continuous
SAAS model.

## Decision

Do **not** add a `MixedSaasSingleTaskGP` wrapper at this time.

BoTorch's mixed model and its MAP-SAAS models encode different kernel structures:

- `MixedSingleTaskGP` constructs categorical/continuous additive and interaction
  kernels and accepts a continuous-kernel factory.
- `AdditiveMapSaasSingleTaskGP` and `EnsembleMapSaasSingleTaskGP` implement
  dedicated MAP-SAAS models for continuous inputs.

Simply injecting a SAAS-flavoured continuous kernel into `MixedSingleTaskGP`
would not reproduce the MAP-SAAS model semantics, especially the tau handling
and ensemble structure. Naming such a composition `MixedSaasSingleTaskGP`
would therefore overstate what the model implements.

## Current recommendation

For mixed high-dimensional problems:

1. use `MixedSingleTaskGP` when categorical structure is essential;
2. use the mixed reduced models when dimensionality reduction is appropriate;
3. use `AdditiveMapSaasSingleTaskGP` or `EnsembleMapSaasSingleTaskGP` only
   when all modeled dimensions are continuous.

A future mixed SAAS model should be introduced only with a dedicated kernel and
inference design plus benchmark evidence. It must not be a compatibility alias
or a renamed `MixedSingleTaskGP`.
