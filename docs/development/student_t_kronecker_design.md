# Student-t x Kronecker design audit

## Decision

Phase 13 keeps Student-t x Kronecker at **Prototype / variational-architecture gated**.

A Student-t observation model is statistically meaningful for aligned multi-task data, but it is
not an exact-Gaussian Kronecker extension. The non-Gaussian likelihood breaks the closed-form
posterior used by `KroneckerMultiTaskGP`. A valid implementation therefore needs a dedicated
block-design variational latent process and a multi-output Student-t observation contract.

The existing long-format `StudentTMultiTaskGP` is useful evidence for heavy-tailed multi-task
modeling, but converting `train_Y[n, m]` to `nm × 1` rows would abandon the Kronecker
block-design contract. Phase 13 does not add such a wrapper.

## Target statistical model

Training data remain

- `train_X[..., n, d]`;
- `train_Y[..., n, m]`.

The latent process should preserve separable covariance

`f ~ GP(mu, K_data ⊗ K_task)`.

For each aligned observation-task cell,

`y[i, t] | f[i, t] ~ StudentT(df, loc=f[i, t], scale=sigma_t)`.

The first public design should use a shared degrees-of-freedom parameter `df` and either a shared
scale or explicit task-specific scales. Task-specific `df` is deferred because it adds
identifiability and optimization complexity without being necessary to establish the architecture.

`df > 2` remains the default public constraint so observation variance is finite.

## Why exact Kronecker inference is unavailable

With Gaussian observations, the block-design model can exploit the Kronecker latent covariance in
an exact Gaussian posterior. A Student-t likelihood is non-conjugate:

`p(f | y) proportional to p(y | f) p(f)`

is no longer Gaussian.

Replacing the Gaussian likelihood on the existing exact `KroneckerMultiTaskGP` does not create a
valid posterior implementation. The model requires approximate inference.

## Variational latent contract

A future implementation should own a block-aware variational model rather than reuse the current
long-format `SingleTaskVariationalGP` wrapper unchanged.

At minimum it must define:

- inducing representation for the data axis;
- representation of the task covariance factor;
- a variational distribution that preserves the output-task axis;
- KL divergence against the intended Kronecker latent prior;
- expected Student-t log likelihood over all `n × m` observation cells;
- posterior conversion to a BoTorch-compatible multi-output posterior.

A naive independent variational GP per task is rejected because it removes `K_task`.

## Inducing-point alternatives

### Data-axis inducing points with task covariance

Use `Z[u, d]` on the data axis and retain an explicit task covariance for the inducing outputs.
This is the preferred first prototype because task identity remains an output dimension.

The inducing latent state is conceptually `u × m`, with covariance structured by inducing-data
and task factors.

### Flattened inducing points

Appending a task index and using `um × (d + 1)` inducing rows is a long-format model. It may be
useful independently but is not Student-t x Kronecker under this development contract.

Phase 13 rejects it as the implementation shortcut.

## Likelihood semantics

GPyTorch's scalar `StudentTLikelihood` is sufficient for the existing scalar and long-format
models because each training row corresponds to one scalar observation.

For block-design outputs, the future likelihood path must make task broadcasting explicit. A
single shared likelihood scale may be supported deliberately, but accidental broadcasting from a
scalar implementation must not be presented as task-specific heavy-tailed noise.

If task-specific scales are implemented, their shape and mapping to the `m` output tasks must be
tested directly.

## Training objective

Like the existing Student-t family, a future block-design model should expose
`supports_mll = False` unless it implements a repository-standard variational objective wrapper.

The core objective is an ELBO-style loss:

`loss = -E_q(f)[log p(y | f)] + KL(q(f) || p(f))`.

The expected log likelihood must sum or reduce over both observation and task axes exactly once.
The MCMC/sample batch, candidate batch, observation, and task dimensions must not be conflated.

A `training_loss()` API is preferred initially because that matches the current Student-t
contract and does not falsely advertise exact MLL fitting.

## Posterior contract

After variational fitting, `posterior(X)` must return a BoTorch-compatible posterior with mean
shape

`... × q × m`

and samples

`sample_shape × ... × q × m`.

The posterior used by BO is the approximate latent GP posterior. The heavy-tailed observation
likelihood is part of training and optional observation prediction; acquisition functions should
not receive an extra likelihood dimension.

## Runtime acceptance tests

Before public registration, a future prototype must prove:

1. raw block-design X/Y are retained without a task column;
2. the latent covariance / variational prior includes nontrivial task covariance;
3. one training step gives finite loss and finite gradients;
4. `df > 2` validation is enforced;
5. posterior mean and samples preserve the `m` task axis;
6. task covariance affects cross-task posterior behavior;
7. no implicit flattening to long-format occurs.

A two-task synthetic problem with correlated latent functions and injected heavy-tailed residuals
is the minimum useful test.

## Acquisition and optimizer gate

Public promotion additionally requires runtime evidence for:

- scalarized MC `qLogExpectedImprovement`;
- `qUpperConfidenceBound`;
- standard BoTorch posterior sampling;
- continuous `optimize_acqf`.

Analytic acquisitions should not be assumed compatible with a variational multi-output posterior.

## Mixed extension

Mixed Student-t x Kronecker is deferred until the continuous block-design variational architecture
works. The Mixed version should replace only the data covariance with the established mixed
continuous/categorical covariance. `cat_dims` remains raw data coordinates; task identity remains
the output axis.

## Public API decision

Phase 13 does not add:

- `StudentTKroneckerMultiTaskGP`;
- a Mixed counterpart;
- registry/capability metadata;
- generated model coverage;
- aliases to `StudentTMultiTaskGP`.

The existing long-format Student-t multi-task model remains a separate valid model and must not be
described as Kronecker support.

## Revisit trigger

Move from Prototype to Implement when robotorchan has a small, maintainable block-design
variational GP primitive that can represent `K_data ⊗ K_task`, compute its KL term, and expose a
BoTorch-compatible `q × m` posterior without relying on long-format reshaping.

This is an inference-architecture blocker, not a statistical objection to Student-t observations.
