# Heteroskedastic Kronecker MultiTask design

## Decision

Phase 6 accepts a task-specific heteroskedastic Kronecker model for implementation in Phase 7.
The response remains a block-design Kronecker GP. Observation noise is represented explicitly as
an `[..., n, m]` matrix and is never silently broadcast from `[..., n, 1]`.

The existing long-format `HeteroskedasticMultiTaskGP` cannot be reused by reshaping the outputs:
that would replace the block-design posterior contract with a task-feature model.

## Public contract

The planned model is `HeteroskedasticKroneckerMultiTaskGP` with:

- `train_X[..., n, d]` containing data features only;
- `train_Y[..., n, m]` containing aligned task outputs;
- `noise_floor > 0`;
- optional explicit initial `train_Yvar[..., n, m]`;
- task-specific noise as the default and only Phase 7 public mode;
- no `task_feature` argument.

`raw_train_X`, `raw_train_Y`, and `raw_train_Yvar` preserve caller coordinates. If
`train_Yvar` is omitted, `raw_train_Yvar` records the initialized `noise_floor` matrix rather
than pretending that no observation model exists.

## Observation covariance

Let `f(X)` have latent Kronecker covariance

`K_f = K_data(X, X') ⊗ K_task`.

For aligned training observations, Phase 7 uses a task-specific diagonal observation variance

`D_noise = diag(vec(V(X)))`, where `V(X)` has shape `[..., n, m]`.

The observation covariance is therefore

`K_y = K_f + D_noise`.

This is deliberately not `K_data ⊗ K_task + sigma(X)^2 I` with one scalar noise process shared
across tasks. Different measured properties may have different noise at the same design point.

A runtime acceptance test must change one task's learned variance while holding the latent model
fixed and demonstrate that the corresponding observation covariance changes without changing the
other task's requested variance.

## Noise surrogate

Phase 7 should use one block-design noise surrogate whose outputs are the `m` task-specific log
variances. The preferred first implementation is another `KroneckerMultiTaskGP` trained on

`log(residual**2 + noise_floor)`.

This preserves the output axis and can share statistical strength between task-specific noise
surfaces. It also avoids constructing `m` unrelated `SingleTaskGP` objects.

The iterative fitting lifecycle is:

1. fit the response MLL;
2. compute latent response residuals at the aligned training design;
3. fit the block-design log-noise GP on the `[..., n, m]` residual variance targets;
4. predict an `[..., n, m]` variance matrix;
5. update the response observation-noise operator;
6. invalidate prediction caches before the next response fit.

`predicted_noise(X)` returns `[..., q, m]`. `noise_posterior(X)` returns the noise model's
multi-output posterior.

## Likelihood boundary

Upstream `KroneckerMultiTaskGP` currently owns a `MultitaskGaussianLikelihood`, whose learned
noise parameters do not provide an input-dependent `n × m` fixed-noise matrix. Therefore Phase 7
must not claim heteroskedasticity merely by changing `task_noises` or a scalar `noise`.

Implementation should introduce the smallest dedicated likelihood / observation-noise path needed
to add `diag(vec(V(X)))` at observed points while preserving the latent Kronecker posterior.
If this cannot be done without copying substantial private BoTorch posterior logic, Phase 7 must
remain a prototype rather than expose a misleading public model.

## Posterior semantics

`posterior(X, observation_noise=False)` is the latent function posterior.

For `observation_noise=True`, the model must use `predicted_noise(X)` with shape
`[..., q, m]`. The observation posterior must therefore differ from the latent posterior by the
predicted task-specific diagonal variance.

Training-time likelihood noise and prediction-time observation noise must have the same task
ordering as columns of `train_Y`.

## MLL lifecycle

The response model remains exact Gaussian conditional on the current fixed noise matrix, so
`make_mll()` remains an exact MLL for each alternating response-fit step. The noise surrogate has
its own exact MLL. The alternating procedure is not presented as joint maximum likelihood.

The implementation must reset relevant prediction strategies after changing the fixed noise matrix;
otherwise posterior calls may use stale covariance caches.

## Shape and validation rules

Phase 7 must reject:

- `train_X.ndim < 2`;
- `train_Y.ndim < 2`;
- mismatched observation dimension `n`;
- `train_Yvar` whose shape is not exactly compatible with `train_Y`;
- non-positive `noise_floor`;
- negative or non-finite explicit variances.

No implicit task-axis broadcasting is allowed.

## Acquisition and sampling acceptance

Before public registration, Phase 7 must demonstrate:

- finite latent posterior mean/variance;
- finite posterior samples preserving the `m` output axis;
- finite task-specific `predicted_noise(X)`;
- finite observation posterior with task-specific variance contribution;
- scalarized `qLogExpectedImprovement`;
- continuous `optimize_acqf`;
- common `make_mll()` behavior.

Multi-output acquisition support is tested only where the acquisition contract itself is compatible;
registration must not imply qEHVI/qNEHVI compatibility without an explicit runtime test.

## Mixed boundary

Mixed heteroskedastic Kronecker is deferred to Phase 8. It must use mixed covariance in both the
response data factor and the noise surrogate. Categorical coordinates remain data features only;
the output task axis is never listed in `cat_dims`.

## Phase 7 go / hold criterion

Proceed with a public Phase 7 model only if the implementation can satisfy all three conditions:

1. learned noise has explicit `n × m` semantics;
2. observation covariance demonstrably responds task-by-task to that noise;
3. the implementation preserves BoTorch posterior/acquisition behavior without a large copy of
   private upstream Kronecker internals.

If any condition fails, keep the work as a documented prototype and record the blocking upstream
contract instead of shipping a pseudo-heteroskedastic model.
