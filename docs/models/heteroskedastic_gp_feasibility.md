# Heteroskedastic GP feasibility

## Decision

Phase 6 does not add a nominal `HeteroskedasticSingleTaskGP` wrapper on top of
BoTorch 0.18.1 because the current dependency does not expose a maintained
heteroskedastic exact-GP model that matches robotorchan's wrapper policy.

A fake class implemented by silently converting learned noise into fixed
`train_Yvar` would not be a heteroskedastic GP. It would freeze observation
variance rather than jointly model input-dependent uncertainty.

## Required statistical contract

A future public heteroskedastic surrogate must represent

```text
y = f(x) + epsilon(x)
epsilon(x) ~ N(0, sigma(x)^2)
```

where the noise process depends on `x`. The implementation must expose a
BoTorch-compatible posterior and a training objective that trains the intended
noise model rather than treating per-observation variance as immutable data.

## What is already supported

Known per-observation variances remain supported through existing
`train_Yvar` arguments where the underlying BoTorch model accepts them. This
is fixed-noise modeling and must not be documented as heteroskedastic learning.

The robust scenario and risk layers from Phases 3-5 are orthogonal to this
surrogate concern and remain composable with ordinary models.

## Implementation options

A later implementation may proceed when one of these contracts is selected and
tested:

1. a maintained upstream BoTorch heteroskedastic model becomes available;
2. a robotorchan variational two-process model is implemented explicitly;
3. a custom likelihood / latent-noise construction is implemented with a clear
   posterior and training contract.

Option 2 is the preferred independent implementation path if heteroskedastic
learning is required before upstream support exists.

## API boundary

The intended public name remains `HeteroskedasticSingleTaskGP`, but Phase 6
does not reserve that name with a placeholder class. The repository rule against
compatibility and placeholder APIs takes precedence.

When implemented, the model should follow the same raw-training-data and
training-API conventions as the existing model family. It must not masquerade
as `ExactGPModelMixin` if its inference is variational or otherwise non-exact.

## Mixed and high-dimensional boundaries

Mixed, reduced, and multi-task heteroskedastic variants are intentionally not
created before the base statistical model is validated. Once the base model
exists, composition should be preferred where reduction or scenario handling
does not change the noise-process mathematics.
