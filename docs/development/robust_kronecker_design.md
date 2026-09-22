# Robust Kronecker MultiTask design audit

This Phase 5 audit decides which robust / noise families should receive block-design Kronecker
variants. It does not treat every long-format MultiTask model as requiring a Kronecker twin.

## Decision criteria

A Kronecker variant is accepted only when all of the following are true:

1. observations are naturally represented as aligned `train_Y[..., n, m]`;
2. task identity belongs to the output axis rather than an input `task_feature`;
3. the family-specific observation model has a clear task-sharing parameterization;
4. BoTorch posterior and sampling contracts can be preserved without flattening to long format;
5. the variant has a practical use case distinct from the existing long-format model.

Mixed input is a second decision. A robust Kronecker model is not automatically given a Mixed
variant unless categorical data covariance can remain independent of output-task covariance.

## Family decisions

| Family | Kronecker decision | Mixed Kronecker | Priority | Reason |
| --- | --- | --- | --- | --- |
| Robust Relevance Pursuit | implement | implement after reference path | high | Sparse observation outliers are meaningful for aligned multi-output experiments; use as reference implementation. |
| Heteroskedastic | implement | evaluate after continuous version | high | Input-dependent noise is common in block-design measurements, but task-specific noise semantics must be explicit. |
| Nonstationary | implement | evaluate after continuous version | high | Nonstationary data covariance is orthogonal to the output task covariance and has clear practical value. |
| Student-t | defer | defer | medium | Heavy-tail likelihood is meaningful, but the current implementation is variational and long-format; a block-design variational posterior needs a dedicated design rather than a wrapper. |
| Contaminated | defer | defer | medium | The contamination mixture is useful, but task-shared versus task-specific contamination parameters must be chosen explicitly first. |
| Replicate noise | defer | defer | low | Replicate grouping across an aligned output matrix needs a separate grouping contract. |
| Joint heteroskedastic | do not implement in first wave | no | low | Joint latent response/noise structure should be settled before adding output-axis Kronecker structure. |

## First-wave public models

Phase 6 should start with:

- `RobustRelevancePursuitKroneckerMultiTaskGP`;
- `HeteroskedasticKroneckerMultiTaskGP`;
- `NonstationaryKroneckerMultiTaskGP`.

The first Mixed candidate is:

- `MixedRobustRelevancePursuitKroneckerMultiTaskGP`.

Mixed heteroskedastic and nonstationary variants should only be added after the continuous
Kronecker implementations demonstrate that data covariance, task covariance, and observation
noise remain cleanly separated.

## Input and output contract

Kronecker robust models use the same block-design surface as `KroneckerMultiTaskGP`:

- `train_X[..., n, d]` contains data features only;
- `train_Y[..., n, m]` contains the aligned task outputs;
- there is no `task_feature` argument;
- `raw_train_X` and `raw_train_Y` preserve caller-supplied tensors;
- output tasks must not be flattened into long-format rows internally merely to reuse a
  `MultiTaskGP` implementation.

For Mixed variants, `cat_dims` indexes data features in `train_X`. Output tasks are never
categorical dimensions.

## Observation-model decisions

### Robust relevance pursuit

Use block-design response covariance from the Kronecker model while applying relevance-pursuit
observation-noise semantics to aligned observations. The implementation must define whether the
support is per observation-output cell or shared across outputs at an input location. Phase 6
should prefer the representation supported naturally by the underlying likelihood and must expose
that choice in the docstring and tests.

### Heteroskedastic

The response model remains block-design. The noise model must predict observation variance with an
explicit output-task axis. A single scalar noise prediction silently broadcast over all tasks is
not acceptable unless the public API explicitly requests shared noise.

The first implementation should use task-specific noise by default because it preserves the
practical distinction between differently noisy measured properties.

### Nonstationary

Nonstationarity belongs to the input/data covariance. Task covariance remains the Kronecker output
factor. This is the cleanest robust/noise Kronecker extension after the reference model and should
not introduce task-dependent length-scale behavior unless separately designed.

## Runtime acceptance criteria

Each first-wave model must prove:

1. block-design input/output shape validation;
2. raw training-data preservation;
3. finite posterior mean and variance;
4. finite posterior samples with the declared sampler;
5. output dimension is preserved in posterior samples;
6. one representative scalarized MC acquisition;
7. one representative multi-output acquisition when compatible;
8. `make_mll()` behavior matches the actual inference contract.

Mixed variants additionally require negative `cat_dims`, categorical-kernel path evidence, and
validation that categorical dimensions refer only to `train_X`.

## Explicit non-goals

Phase 6 must not:

- create Kronecker variants for every robust family;
- emulate Kronecker behavior by reshaping long-format MultiTask data;
- add aliases for long-format models;
- add Student-t or contaminated Kronecker models before their block-design variational /
  likelihood semantics are designed;
- claim acquisition compatibility from model registration alone.

## Follow-up boundary

After the first-wave runtime tests pass, revisit Student-t and contaminated models using evidence
from the implemented Kronecker posterior contract. They remain valid future candidates, not
rejected model families.
