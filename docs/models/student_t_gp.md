# Phase 7: heavy-tailed observation model feasibility

## Purpose

Phase 7 addresses observation robustness that is not covered by relevance pursuit
or heteroskedastic Gaussian noise: heavy-tailed residuals and isolated large
errors that should be modeled through the observation likelihood itself.

The target statistical model is

```text
f(x) ~ GP
y | f(x) ~ StudentT(df=nu, loc=f(x), scale=sigma)
```

with `nu > 2` as the default finite-variance regime.

## Why this is a separate surrogate

A Student-t observation model changes the likelihood and inference procedure.
It is therefore a genuine surrogate-model concern rather than an acquisition
risk measure or a `Robust*` composition wrapper.

It is also distinct from the existing models:

- relevance pursuit models sparse gross outliers;
- heteroskedastic models input-dependent Gaussian variance;
- Student-t likelihood models globally heavy-tailed observation residuals.

All three should remain available because their assumptions differ.

## Inference decision

Do not force Student-t observations into an exact Gaussian MLL. The Phase 7
implementation should use variational GP inference and expose a
`training_loss()` contract, following the same non-exact-inference principle as
`JointHeteroskedasticSingleTaskGP`.

The first implementation should be scalar-output and continuous-input only.
Mixed, reduced-input, multi-task, and heteroskedastic-Student-t cross products
are deferred until the base inference model is validated.

## Public API target

```python
model = StudentTSingleTaskGP(
    train_X,
    train_Y,
    num_inducing=32,
    df=4.0,
)

loss = model.training_loss()
posterior = model.posterior(test_X)
```

Required behavior:

- retain raw training data;
- `supports_mll = False`;
- `make_mll()` must not claim an exact Gaussian MLL;
- provide a BoTorch-compatible latent response posterior;
- preserve dtype/device migration and state serialization;
- support MC acquisition functions that consume the latent posterior.

Whether `df` and observation scale are fixed or learned must be explicit in
the constructor and state dictionary. The initial implementation should prefer
a stable constrained parameterization over implicit magic defaults.

## Validation gates

Before Phase 7 is complete, tests must cover:

1. finite differentiable training loss;
2. gradients through variational GP parameters and Student-t likelihood
   parameters that are configured as learnable;
3. finite posterior with batch and q dimensions;
4. state-dict round trip in evaluation mode;
5. float32/float64 migration;
6. MC acquisition compatibility;
7. robustness smoke test showing that one extreme response produces less
   likelihood penalty than under a comparable Gaussian observation model;
8. no regression to the two heteroskedastic models or relevance-pursuit models.

## Composition boundary

Input perturbation, environmental scenarios, CVaR, worst-case, and SN ratio stay
outside this model. They compose with the posterior/acquisition layer exactly as
specified by the robust architecture.

Do not introduce `RobustStudentTGP`, `StudentTPCAGP`,
`StudentTHeteroskedasticGP`, or similar cross-product classes in this phase.

## Phase 7 implementation sequence

1. Audit current GPyTorch Student-t likelihood semantics and the existing
   robotorchan variational wrapper.
2. Implement the scalar continuous `StudentTSingleTaskGP`.
3. Add training, lifecycle, dtype, posterior, and acquisition tests.
4. Add an outlier-resistance likelihood test against Gaussian residual penalty.
5. Register the public model and non-MLL contract atomically.
6. Run stale-contract, Ruff lint/format, Python 3.11/3.12/3.13, package, and
   notebook gates before merge.
