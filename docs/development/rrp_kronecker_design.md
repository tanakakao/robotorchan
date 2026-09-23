# Robust relevance pursuit x Kronecker design audit

## Decision

Phase 12 keeps Robust Relevance Pursuit (RRP) x Kronecker at **Prototype / research-gated**.

The existing robotorchan RRP paths do not provide a valid composition primitive for block-design
Kronecker outputs:

- the single-task implementation delegates to BoTorch's dedicated RRP single-task model;
- the current multi-task implementation applies `RobustRelevancePursuitMixin` to a long-format
  `MultiTaskGP`;
- neither contract defines sparse outlier support over an aligned `n × m` block-design response.

Reusing the single-task `noise_covar` machinery or flattening the block into long-format rows
would change the model semantics. Phase 12 therefore does not add a public runtime class.

## Target block-design model

Training data remain

- `train_X[..., n, d]`;
- `train_Y[..., n, m]`.

The latent response covariance remains

`K_f = K_data(X, X') ⊗ K_task`.

RRP should augment the observation model with sparse outlier corrections on the aligned
observation-task grid, not modify the data or task covariance factors.

A conceptual observation model is

`y = f + epsilon + delta`

where `delta[n, m]` is sparse and `f` retains Kronecker covariance.

## Support semantics

Three support structures are statistically plausible but are not interchangeable.

### Observation-task support

Each cell `(i, t)` can be selected independently. This is the most general interpretation and
allows one task to be corrupted while other tasks at the same design point remain clean.

Support size is `n × m`.

### Shared-row support

A selected design row marks all tasks at that X as contaminated. This is appropriate only when an
outlier source is known to affect the whole measurement event.

Support size is `n`, broadcast deliberately across tasks.

### Structured task support

A hierarchical model could combine row-level and task-specific sparsity. This is expressive but
requires a new structured relevance-pursuit algorithm and is not justified as the first
implementation.

Phase 12 chooses **observation-task support** as the default target for any future public model.
Shared-row support may be added only as an explicit option, never as an implicit shortcut.

## Why the current RRP mixin is not reusable directly

The existing long-format `RobustRelevancePursuitMultiTaskGP` initializes
`RobustRelevancePursuitMixin` with the number of long-format training rows. Its sparse noise
support therefore follows that representation.

A Kronecker block has `n` design rows but `n × m` observation cells. Passing `n` to the
single-task-style mixin would create row-level support, while reshaping to `nm` rows would abandon
the block-design model.

The required support operator must understand the task axis explicitly.

## Required sparse observation operator

A future implementation needs a block-aware robust observation component with at least:

- support indices represented in observation-task coordinates;
- sparse correction/noise parameters with shape compatible with `n × m`;
- deterministic mapping from `(observation, task)` to the flattened ordering used by the
  Kronecker observation covariance;
- support expansion and pruning that preserve this mapping;
- a likelihood or marginal-likelihood path that combines sparse corrections with the Kronecker
  latent covariance.

This component should be reusable by continuous and Mixed Kronecker response models.

## MLL and fitting lifecycle

RRP is not just a covariance-kernel substitution. Public implementation requires an optimization
lifecycle that alternates or coordinates:

1. exact-GP hyperparameter fitting for the Kronecker latent model;
2. relevance-pursuit support selection;
3. optimization of sparse outlier/noise parameters;
4. cache invalidation after support changes.

The current `make_mll` contract must remain truthful. A model must not advertise ordinary exact
MLL fitting if additional relevance-pursuit steps are required but not executed.

## Posterior semantics

The public posterior must distinguish latent prediction from robust observation handling.
Conditioning on the selected sparse support must influence the fitted latent posterior, while
posterior queries at new X must not invent an outlier support index for unseen candidate points.

A future implementation must document whether `observation_noise=True` includes only ordinary
measurement noise or any predictive contamination model. RRP's training-point sparse corrections
must not be silently interpreted as input-dependent future noise.

## Runtime acceptance tests

Before public registration, executable tests must prove:

1. one corrupted task cell can be selected without forcing all tasks in that row into support;
2. support flattening matches the Kronecker task ordering;
3. changing a sparse correction changes the intended observation covariance / fitted response only;
4. latent posterior and samples remain finite with shape `... × q × m`;
5. clean data do not create pathological dense support;
6. `make_mll` and the relevance-pursuit fitting lifecycle are internally consistent.

A synthetic two-task case with a single task-specific outlier is the minimum useful regression
test.

## Acquisition and optimizer gate

After robust fitting is valid, public promotion additionally requires:

- scalarized MC `qLogExpectedImprovement`;
- `qUpperConfidenceBound`;
- posterior sampling;
- continuous `optimize_acqf`.

Acquisition functions operate on the fitted latent response posterior; sparse training support is
not itself an acquisition output dimension.

## Mixed extension

Mixed RRP x Kronecker is deferred until the continuous block-aware RRP operator exists. A future
Mixed variant changes only the data covariance factor. The robust support remains indexed by
observation row and output task, and `cat_dims` must never include task identity.

## Public API decision

Phase 12 does not add:

- `RobustRelevancePursuitKroneckerMultiTaskGP`;
- a Mixed counterpart;
- registry or capability metadata;
- generated coverage entries;
- aliases to the long-format RRP model.

The item remains research-gated until a dedicated block-aware sparse observation operator can be
implemented without copying substantial private BoTorch RRP internals.

## Revisit trigger

Move from Prototype to Implement when either:

1. BoTorch exposes an RRP component whose support dimension can explicitly represent block-design
   observation-task cells; or
2. robotorchan deliberately accepts ownership of a small, independently testable block-aware RRP
   observation operator with a stable MLL/fitting contract.

This is a model-architecture blocker, not evidence against relevance pursuit for multitask data.
