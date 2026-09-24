# Contaminated Gaussian x Kronecker design audit

## Decision

Phase 14 keeps Contaminated x Kronecker at **Prototype / variational-architecture gated**.

The contamination likelihood is statistically meaningful for aligned multi-task data, but it is a
non-Gaussian mixture likelihood. As with Student-t observations, the exact Gaussian posterior of
`KroneckerMultiTaskGP` is no longer available. A valid implementation needs the same
block-design variational latent primitive identified in Phase 13.

Phase 14 resolves the previously open contamination-parameter question: the first public design
should use **shared mixture probability with task-specific component scales**. Shared parameters
remain a simpler explicit option; fully task-specific mixture probabilities are deferred.

## Target statistical model

Training data remain

- `train_X[..., n, d]`;
- `train_Y[..., n, m]`.

The latent process preserves

`f ~ GP(mu, K_data ⊗ K_task)`.

For observation `(i, t)`:

`y[i,t] | f[i,t] ~ (1-p) N(f[i,t], sigma_in,t^2) + p N(f[i,t], sigma_out,t^2)`

with

`0 < p < 1`

and

`0 < sigma_in,t < sigma_out,t`.

The contamination indicator is observation-task local even when the mixture probability `p` is
shared.

## Parameter-sharing decision

### Default target: shared p, task-specific scales

This is the preferred first public contract.

A shared contamination probability avoids introducing `m` weakly identified mixture weights,
while task-specific inlier/outlier scales allow tasks with different physical noise magnitudes to
retain appropriate residual units.

Shapes are conceptually:

- `p`: scalar;
- `sigma_in`: `m`;
- `sigma_out`: `m`.

### Optional shared-all mode

A single inlier scale and outlier scale may be supported explicitly for tasks already standardized
to comparable noise levels. This must be a named/explicit parameterization, not accidental
broadcasting.

### Deferred: task-specific p

A vector `p[m]` is statistically possible, but mixture probability and outlier scale can trade off
strongly with limited data. It is deferred until the simpler model has runtime evidence.

## Why exact Kronecker inference is unavailable

The Gaussian mixture likelihood is non-conjugate. Therefore a covariance of
`K_data ⊗ K_task` does not imply an exact Gaussian posterior after conditioning on contaminated
observations.

The existing exact `KroneckerMultiTaskGP` likelihood cannot simply be replaced by the current
contamination loss. Approximate inference is required.

## Relationship to Phase 13

Student-t and contaminated observations need the same latent infrastructure:

- block-design `n × m` targets;
- data-axis inducing points;
- explicit task covariance;
- a variational distribution preserving the task axis;
- KL against the Kronecker latent prior;
- BoTorch-compatible `q × m` posterior.

The likelihood expectation differs, but the variational latent process should be shared. Phase 14
therefore should not create a second independent variational Kronecker implementation.

## Training objective

The current contaminated family uses Monte Carlo samples from the latent variational distribution
to estimate mixture expected log likelihood. A block-design implementation should retain that
approach:

`loss = -E_q(f)[log p_mix(y | f)] + beta * KL(q(f) || p(f))`.

The likelihood term reduces over both observation and task axes. `beta > 0` remains explicit.

The public model should use `supports_mll = False` and `training_loss()` initially, matching the
existing contamination and Student-t families.

## Contamination diagnostic

A future model should expose a diagnostic with output shape `... × q × m` giving the approximate
posterior probability that each supplied observation belongs to the contamination component.

For task `t`, the diagnostic compares

- `(1-p) N(residual; 0, sigma_in,t^2)`;
- `p N(residual; 0, sigma_out,t^2)`.

This diagnostic is not the latent GP posterior and must not be registered as an acquisition
objective automatically.

## Runtime acceptance tests

Before public registration, tests must prove:

1. raw block-design X/Y remain unflattened;
2. task covariance is nontrivial and affects cross-task predictions;
3. shared `p` and task-specific scales have the intended shapes;
4. invalid probabilities and `sigma_out <= sigma_in` are rejected task by task;
5. one training step gives finite Monte Carlo loss and gradients;
6. posterior mean/samples preserve `... × q × m`;
7. contamination diagnostics return one value per observation-task cell;
8. increasing one task's outlier scale does not silently change another task's scale.

## Acquisition and optimizer gate

Public promotion requires runtime evidence for:

- scalarized MC `qLogExpectedImprovement`;
- `qUpperConfidenceBound`;
- posterior sampling;
- continuous `optimize_acqf`.

BO acquisitions consume the fitted latent response posterior, not the contamination diagnostic.

## Mixed extension

Mixed Contaminated x Kronecker is deferred until the shared block-design variational primitive
exists. Mixed treatment changes only `K_data`; task covariance and contamination parameters remain
indexed by the output task axis. `cat_dims` must remain raw data coordinates.

## Public API decision

Phase 14 does not add:

- `ContaminatedKroneckerMultiTaskGP`;
- a Mixed counterpart;
- registry/capability metadata;
- generated coverage;
- aliases to the existing long-format `ContaminatedMultiTaskGP`.

The long-format model remains valid independently but is not Kronecker block-design support.

## Revisit trigger

Move from Prototype to Implement after the Phase 13/14 shared variational Kronecker primitive can:

1. represent data-axis inducing variables with explicit task covariance;
2. compute a stable KL term;
3. evaluate observation-task likelihood terms without flattening task identity into X;
4. expose a BoTorch-compatible multi-output posterior.

At that point Student-t and contaminated likelihoods should be implemented as two observation
heads over the same latent architecture rather than as duplicate model stacks.
