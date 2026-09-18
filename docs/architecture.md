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
- `raw_data_names`: ordered names of constructor-level raw-data snapshots;
- `raw_data`: mapping of every raw tensor retained by the wrapper;
- `supports_mll`: whether `make_mll()` is a valid training-objective factory;
- `make_mll()`: construct the correct marginal-likelihood-style objective when supported.

### Raw-data lifecycle

`raw_*` values are provenance snapshots of the tensors supplied to the wrapper constructor. They are **not** a second copy of BoTorch's mutable current training state.

Consequently, model-update operations such as `condition_on_observations()` and `fantasize()` are delegated to BoTorch and do not cause robotorchan to rewrite the raw provenance buffers. A conditioned or fantasy model therefore retains the constructor snapshot from the source wrapper while native BoTorch attributes such as `train_inputs` and `train_targets` describe the updated training state.

This distinction is deliberate. Updating raw buffers inside conditioning or fantasy logic would require robotorchan to override upstream lifecycle methods and would risk changing model semantics. Users who need the current training state should use the native BoTorch attributes; users who need the original caller input should use `raw_*` / `raw_data`.

Do not force supervised names onto models whose data have different semantics. For example, preference models should use names such as `raw_datapoints` and `raw_comparisons` rather than pretending comparisons are ordinary `train_Y` values.

Composite models follow the same principle. `ModelListGP` is not treated as if it had one global training tensor pair. Raw data remains owned by the child models and the wrapper exposes grouped `raw_train_Xs`, `raw_train_Ys`, and `raw_train_Yvars` convenience properties. Its training objective is `SumMarginalLogLikelihood`, not `ExactMarginalLogLikelihood` for the container as a whole.

Variational models also require model-family-specific training semantics. `SingleTaskVariationalGP` retains `raw_train_X` and optional `raw_train_Y`, exposes `raw_train_Yvar = None` because the upstream constructor has no `train_Yvar` argument, and constructs `VariationalELBO` against the internal approximate GP (`model.model`). `make_mll(num_data=None)` defaults to the row count of `raw_train_X`; callers performing minibatch training must pass the total data-set size explicitly.

Preference models use their own raw-data vocabulary. `PairwiseGP` retains `raw_datapoints` and `raw_comparisons` before BoTorch input transforms or duplicate consolidation, preserves the upstream prior-only mode where either value may be `None`, and constructs `PairwiseLaplaceMarginalLogLikelihood`. It deliberately does not expose `raw_train_Y`, because pairwise comparisons are not ordinary supervised targets.

Fully Bayesian models retain the normal supervised raw-data vocabulary, but do **not** use MLL fitting. `SaasFullyBayesianSingleTaskGP` and `SaasFullyBayesianMultiTaskGP` therefore expose `supports_mll = False`; calling `make_mll()` raises `UnsupportedModelOperationError`. Fitting remains delegated directly to BoTorch's `fit_fully_bayesian_model_nuts`. robotorchan does not add a parallel NUTS fitting API in the model wrapper layer. The optional `fully-bayesian` dependency group installs the JAX / NumPyro dependencies required by BoTorch.

Structured exact GPs retain the exact-GP training objective while preserving model-specific raw tensors and fitting contexts. `HigherOrderGP` retains tensor-valued `raw_train_Y` before BoTorch flattens / standardizes its structured outputs. `LatentKroneckerGP` additionally exposes `raw_train_T`, captured before BoTorch broadcasts task / time coordinates, masks missing outputs, or transforms data. Both use `ExactMarginalLogLikelihood`; specialized solver contexts and optimizers remain explicit BoTorch behavior rather than being hidden by the wrapper.

Specialized BoTorch models follow the same principle: use the normal supervised raw-data contract whenever the model actually receives one global training tensor pair, and preserve model-specific data structure otherwise. Additive / MAP-SAAS, robust relevance-pursuit, hierarchical, and contextual wrappers use the exact-GP contract while leaving their kernels and fitting behavior upstream. `HeterogeneousMTGP` instead receives task-specific tensor lists and therefore exposes grouped `raw_train_Xs`, `raw_train_Ys`, and `raw_train_Yvars`; robotorchan does not invent singular raw tensors after BoTorch embeds those inputs into a common feature space.

## Repository maintenance policy

Repository-wide structural changes are completed atomically. When a module or public API moves, update implementation imports, tests, benchmarks, examples, notebooks, and documentation in the same change, then verify that the removed path or name no longer remains anywhere in the repository. Do not add compatibility modules, deprecated wrappers, legacy aliases, or old-name forwarding APIs; migrate callers completely to the current contract.

Before a structural pull request is considered complete, run both `ruff check .` and `ruff format --check .` in addition to the relevant pytest and notebook checks. Passing lint alone is not evidence that formatting is valid.

Permanent documentation should describe current architecture, behavior, usage, theory, benchmarks, or release procedures. Temporary phase-completion markers, status files, and phase-specific documentation indexes should not remain after their information has been incorporated into permanent documentation.

## Compatibility and CI policy

The wrapper surface is currently supported against BoTorch `>=0.18.1,<0.19`. Constructor parity is part of the public compatibility contract, so a new BoTorch minor line should be enabled only after the wrapper signatures and numerical behavior have been revalidated.

CI follows two layers:

1. the core Python 3.11 / 3.12 / 3.13 matrix installs the normal development dependencies and runs all wrapper tests that require only the core package dependencies;
2. a dedicated Python 3.11 job installs the optional `fully-bayesian` dependency group, smoke-tests JAX / NumPyro availability, and runs the fully Bayesian wrapper tests.

Linux CI installs CPU-only PyTorch before robotorchan dependencies so ordinary pull requests do not download CUDA runtime packages that are unused by the test suite. The workflow intentionally does not restore pip caches: an older cache containing CUDA wheels was several gigabytes and cost more to restore than the lean CPU-only dependency set costs to install. Lint and formatting checks run once on Python 3.11 rather than being duplicated across every matrix entry.

Cross-model tests pin the public model export set and require every public wrapper class to expose the `supports_mll` / `make_mll()` capability contract. This is intended to catch API drift when new wrappers are added.

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

Add `SingleTaskVariationalGP` with raw-data retention and `VariationalELBO` construction, including an explicit total-data-size override for minibatch training.

### Phase 6 — preference wrapper

Add `PairwiseGP` with semantically correct `raw_datapoints` / `raw_comparisons` retention, prior-only compatibility, and `PairwiseLaplaceMarginalLogLikelihood` construction.

### Phase 7 — fully Bayesian wrappers

Add `SaasFullyBayesianSingleTaskGP` and `SaasFullyBayesianMultiTaskGP` with raw supervised data retention, explicit non-MLL training semantics, and direct compatibility with BoTorch's `fit_fully_bayesian_model_nuts`.

### Phase 8 — structured advanced GP wrappers

Add `HigherOrderGP` and `LatentKroneckerGP` with exact-MLL support, tensor-output raw retention, semantically correct `raw_train_T` handling, and upstream numerical parity while keeping BoTorch's specialized solver contexts explicit.

### Phase 9 — specialized BoTorch model wrappers

Add `OrthogonalAdditiveGP`, additive / ensemble MAP-SAAS, Robust Relevance Pursuit, hierarchical single- and multi-task GPs, `HeterogeneousMTGP`, and the contextual `SACGP`, `LCEAGP`, and `LCEMGP` models. Preserve specialized BoTorch kernels and fitting dispatch while applying robotorchan raw-data and MLL conventions only where semantically valid.

### Phase 10 — API cleanup and comprehensive tests

Finalize public exports and capability contracts, define constructor-level raw-data provenance across conditioning / fantasy operations, lock the supported BoTorch compatibility line, and streamline CI without changing predictive model behavior.

### Later extension phases

After the existing-model wrapper surface is complete, continue with robotorchan-specific acquisition functions, active-learning criteria, robust / risk-aware methods, ordinal methods, lookahead methods, and new surrogate-model families.
