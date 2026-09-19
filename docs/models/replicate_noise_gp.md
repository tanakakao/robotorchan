# Phase 11: replicate-aware observation noise

## Purpose

When several observations are measured at the same design condition, their
within-condition variation contains direct information about observation noise.
Phase 11 defines a model path that uses this replicate structure instead of
asking a heteroskedastic GP to infer all noise variation from residuals.

For replicate group `g`:

```text
y_g,r = f(x_g) + epsilon_g,r
epsilon_g,r ~ Normal(0, sigma_g^2)
```

The replicate observations share the same latent response `f(x_g)`, while
`sigma_g^2` is estimated from the repeated measurements.

## Statistical role

This is different from the existing models:

- `HeteroskedasticSingleTaskGP` estimates input-dependent noise indirectly
  through residuals;
- `JointHeteroskedasticSingleTaskGP` learns a latent log-noise process jointly;
- `StudentTSingleTaskGP` models globally heavy-tailed residuals;
- relevance pursuit targets sparse gross outliers.

Replicates provide observed local evidence about noise and should be used when
the experimental design actually contains repeated conditions.

## Initial scope

Phase 11 starts with scalar continuous outputs and exact duplicate design rows.
Approximate grouping by tolerance is intentionally deferred because feature
scales and mixed variables make a universal tolerance unsafe.

Two representations should be supported at the data-preparation boundary:

1. raw replicated observations, grouped by identical `train_X`;
2. explicit group identifiers when callers already know the replicate grouping.

The surrogate should ultimately train on one latent design row per group with:

```text
group_mean = mean(y_g,r)
group_noise = sample_variance(y_g,r) / n_g
```

because the likelihood is attached to the group mean. The distinction between
observation variance `sigma_g^2` and variance of the mean
`sigma_g^2 / n_g` must be explicit.

## Single-observation groups

A group with `n_g = 1` cannot estimate sample variance by itself. The API must
not silently assign zero noise.

The implementation should provide an explicit policy, for example:

- require all groups to contain at least two replicates; or
- accept a caller-supplied fallback variance.

The initial implementation should prefer the strict policy unless a repository
use case requires mixed replicate counts.

## Numerical policy

Estimated noise must be finite and nonnegative. A configurable positive
`noise_floor` may be applied to variance-of-the-mean values for numerical
stability, but the raw empirical variance should remain inspectable.

## Proposed API

A dedicated constructor/factory is preferable to a combinatorial model family:

```python
model = ReplicateNoiseSingleTaskGP.from_replicates(
    train_X,
    train_Y,
    noise_floor=1e-6,
)
```

The resulting surrogate follows the normal exact-GP contract and exposes the
aggregated training data and empirical noise estimates.

This is not a compatibility alias for another model. It represents a distinct
data-generating assumption and preprocessing contract.

## Required tests

1. exact duplicate rows are grouped deterministically;
2. group means are correct;
3. unbiased sample variances are correct;
4. likelihood noise uses variance of the mean, not raw replicate variance;
5. single-observation groups follow the documented strict policy;
6. constant replicates remain numerically valid through `noise_floor`;
7. raw replicate data and group statistics are retained;
8. dtype/device migration and state-dict roundtrip work;
9. posterior batch/q shapes and qMC acquisition smoke tests pass;
10. no behavior of existing heteroskedastic models changes.

## Scope boundaries

Phase 11 does not add:

- fuzzy/tolerance-based grouping;
- correlated noise between replicate observations;
- multitask or mixed cross-product wrappers;
- a learned heteroskedastic residual GP on top of replicate variance;
- automatic pooling/shrinkage of variance estimates.

Those can be evaluated later if data justify them.

## Exit criterion

Phase 11 is complete when replicate-derived observation noise can be used through
a documented exact-GP interface, while preserving the statistical distinction
between replicate variance, variance of the mean, and learned heteroskedastic
noise.
