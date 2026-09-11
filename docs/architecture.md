# Architecture

## Core policy

robotorchan is a **BoTorch extension library**. It should remain interoperable with native BoTorch / PyTorch objects, while allowing thin wrappers around upstream BoTorch models when those wrappers establish a consistent robotorchan API.

Public APIs should therefore use native objects such as `torch.Tensor`, `botorch.models.model.Model`, `AcquisitionFunction`, `PosteriorTransform`, and BoTorch optimizers directly whenever practical.

## Wrapper policy

Wrapping existing BoTorch functionality is allowed when the wrapper adds a cross-model convention rather than merely renaming an upstream class.

Examples of justified wrapper behavior include:

- retaining caller-supplied raw tensors before upstream transforms or preprocessing;
- providing a common `make_mll()` method when the model uses an MLL-style training objective;
- exposing `supports_mll` so fitting semantics are explicit rather than guessed from model type;
- aligning public names and constructor conventions across robotorchan models;
- adding common serialization, metadata, diagnostics, or fitting helpers;
- exposing a stable robotorchan-facing API while delegating predictive behavior to BoTorch.

Wrappers should normally subclass or compose the upstream BoTorch implementation and avoid reimplementing posterior, covariance, transform, conditioning, or fantasy logic that BoTorch already provides.

The first reference implementation is `robotorchan.models.SingleTaskGP`, which subclasses BoTorch `SingleTaskGP` and adds the common wrapper conventions.

## Clean-room development

Implementation in this repository is written from scratch from public papers, public documentation, and public upstream APIs. Source code, tests, internal documentation, naming conventions, or implementation details from prior private/internal projects must not be copied into robotorchan.

When an algorithm is derived from a paper, its implementation should document the reference and any deliberate deviations from the published method.

## Package layout

- `robotorchan.acquisition`: acquisition functions and active-learning criteria.
- `robotorchan.models`: BoTorch-compatible surrogate models and thin wrappers that implement robotorchan model conventions.
- `robotorchan.objectives`: objectives, posterior transforms, and constraint-related helpers when BoTorch does not already provide them.
- `robotorchan.optim`: acquisition optimization and search-space utilities that extend, rather than duplicate, `botorch.optim`.

Additional top-level namespaces should only be introduced when a stable group of functionality exists.

## API rules

1. Prefer subclassing or composing BoTorch abstractions over introducing parallel abstractions.
2. Existing BoTorch models may be wrapped when doing so provides shared robotorchan behavior.
3. Keep upstream constructor arguments and semantics intact unless there is a strong documented reason to differ.
4. Preserve tensor batch semantics used by BoTorch.
5. Avoid application-specific configuration objects in the core package.
6. Do not hide model fitting or acquisition optimization behind implicit global state.
7. New public APIs require tests for shapes, dtype/device behavior, gradients when applicable, and numerical edge cases.
8. Experimental APIs may live under normal modules, but must be clearly documented as unstable until promoted.

## Model wrapper foundation

The model-wrapper layer separates three concerns:

1. **Raw-data retention** — `RawDataMixin` stores arbitrary caller-supplied tensors as detached model buffers. This supports standard supervised names as well as specialized data such as `datapoints`, `comparisons`, or `train_T`.
2. **Model-family conventions** — `SupervisedTrainingDataMixin` defines `raw_train_X`, `raw_train_Y`, and `raw_train_Yvar` for ordinary supervised surrogates.
3. **Training-objective capability** — `ModelTrainingMixin` exposes `supports_mll`. Model families that use a marginal-likelihood objective override `make_mll()` with the appropriate GPyTorch object; unsupported families fail explicitly instead of returning an incorrect objective.

Raw tensors must be snapshotted **before** invoking upstream model initialization whenever that initialization may apply transforms or preprocessing. Stored tensors are detached copies and are registered as buffers when practical, so device / dtype moves and `state_dict` behavior remain natural PyTorch operations.

For supervised model wrappers, use these names consistently where the concept applies:

- `raw_train_X`: caller-supplied training inputs before input transforms;
- `raw_train_Y`: caller-supplied training outcomes before outcome transforms;
- `raw_train_Yvar`: caller-supplied observation variance, or `None`;
- `raw_data`: mapping of every raw tensor retained by the wrapper;
- `supports_mll`: whether `make_mll()` is a valid training-objective factory;
- `make_mll()`: construct the correct marginal-likelihood-style objective when supported.

Do not force supervised names onto models whose data have different semantics. For example, preference models should use names such as `raw_datapoints` and `raw_comparisons` rather than pretending comparisons are ordinary `train_Y` values.

Composite models follow the same principle. `ModelListGP` is not treated as if it had one global training tensor pair. Raw data remains owned by the child models and the wrapper exposes grouped `raw_train_Xs`, `raw_train_Ys`, and `raw_train_Yvars` convenience properties. Its training objective is `SumMarginalLogLikelihood`, not `ExactMarginalLogLikelihood` for the container as a whole.

## Roadmap

### Phase 1 — wrapper foundation

- generic raw-tensor storage;
- supervised raw-data conventions;
- explicit training-objective capability contract;
- exact-GP MLL factory;
- `SingleTaskGP` as the reference wrapper;
- contract tests for raw-data ownership, transforms, dtype/device moves, serialization, and unsupported operations.

### Phase 2 — core exact-GP wrappers

Add `MixedSingleTaskGP` and `SingleTaskMultiFidelityGP` using the Phase 1 contract without reimplementing upstream predictive behavior.

### Phase 3 — multitask wrappers

Add `MultiTaskGP` and `KroneckerMultiTaskGP`, retaining the same supervised raw-data surface where the semantics match.

### Phase 4 — model-list wrapper

Add `ModelListGP` with grouped child raw-data access and `SumMarginalLogLikelihood` construction while preserving native BoTorch child-model interoperability.

### Phase 5 — variational wrapper

Add `SingleTaskVariationalGP` with the appropriate variational training objective rather than forcing exact-GP semantics.

### Later model-wrapper phases

Continue with preference, fully Bayesian, higher-order, latent-Kronecker, and other specialized BoTorch models using model-family-specific training objectives and raw-data conventions.

### Later extension phases

After the existing-model wrapper surface is complete, continue with robotorchan-specific acquisition functions, active-learning criteria, robust / risk-aware methods, ordinal methods, lookahead methods, and new surrogate-model families.
