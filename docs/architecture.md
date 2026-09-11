# Architecture

## Core policy

robotorchan is a **BoTorch extension library**. It should remain interoperable with native BoTorch / PyTorch objects, while allowing thin wrappers around upstream BoTorch models when those wrappers establish a consistent robotorchan API.

Public APIs should therefore use native objects such as `torch.Tensor`, `botorch.models.model.Model`, `AcquisitionFunction`, `PosteriorTransform`, and BoTorch optimizers directly whenever practical.

## Wrapper policy

Wrapping existing BoTorch functionality is allowed when the wrapper adds a cross-model convention rather than merely renaming an upstream class.

Examples of justified wrapper behavior include:

- retaining the raw, pre-transform training tensors supplied by the caller;
- providing a common `make_mll()` method for model-specific marginal-likelihood construction;
- aligning public names and constructor conventions across robotorchan models;
- adding common serialization, metadata, diagnostics, or fitting helpers;
- exposing a stable robotorchan-facing API while delegating predictive behavior to BoTorch.

Wrappers should normally subclass or compose the upstream BoTorch implementation and avoid reimplementing posterior, covariance, transform, or conditioning logic that BoTorch already provides.

The first reference implementation is `robotorchan.models.SingleTaskGP`, which subclasses BoTorch `SingleTaskGP` and adds raw-data retention plus `make_mll()`.

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
2. Existing BoTorch models may be wrapped when doing so provides shared robotorchan behavior such as raw-data retention or `make_mll()`.
3. Keep algorithm configuration explicit in constructors.
4. Preserve tensor batch semantics used by BoTorch.
5. Avoid application-specific configuration objects in the core package.
6. Do not hide model fitting or acquisition optimization behind implicit global state.
7. New public APIs require tests for shapes, dtype/device behavior, gradients when applicable, and numerical edge cases.
8. Experimental APIs may live under normal modules, but must be clearly documented as unstable until promoted.

## Model wrapper conventions

For supervised model wrappers, use these names consistently where the concept applies:

- `raw_train_X`: caller-supplied training inputs before input transforms;
- `raw_train_Y`: caller-supplied training outcomes before outcome transforms;
- `raw_train_Yvar`: caller-supplied observation variance, or `None`;
- `make_mll()`: create the appropriate marginal-likelihood object for the model.

Raw tensors should be retained as detached copies and, where practical, registered as buffers so that device / dtype moves and `state_dict` serialization remain natural PyTorch operations.

A future common fitting helper may consume `model.make_mll()` rather than hard-coding model types.

## Roadmap

### Phase 1 — foundation

- package metadata and `src/` layout;
- linting, tests, and CI;
- architecture and contribution conventions;
- common model-wrapper conventions;
- `SingleTaskGP` reference wrapper with raw-data retention and `make_mll()`.

### Phase 2 — core BoTorch model wrappers

Add thin wrappers only where they provide the same useful common surface, prioritizing models such as `MultiTaskGP`, `KroneckerMultiTaskGP`, `MixedSingleTaskGP`, and `ModelListGP`. Do not force identical methods onto model families when the underlying fitting semantics differ.

### Phase 3 — boundary and uncertainty acquisition functions

Start with small, well-tested analytic criteria useful for active learning and level-set / boundary search, such as Straddle-style acquisition functions. Use BoTorch posterior transforms for multi-output scalarization rather than reinventing scalarization.

### Phase 4 — classification active learning

Add predictive entropy, probability variance, margin-based uncertainty, BALD-style criteria, and boundary-focused methods behind BoTorch-compatible acquisition-function interfaces where the model posterior semantics permit it.

### Phase 5 — multi-objective and constrained extensions

Add methods that are not already available upstream, emphasizing composability with BoTorch objectives, constraints, partitioning, and MC samplers.

### Phase 6 — robust and risk-aware optimization

Add input-uncertainty and risk criteria such as variance / CVaR-oriented extensions when they provide functionality beyond BoTorch's existing risk-measure APIs.

### Phase 7 — ordinal and preference optimization

Add ordinal-target and preference-aware acquisition functions with explicit assumptions and dedicated tests.

### Phase 8 — lookahead and information-theoretic methods

Add computationally heavier multi-step, entropy, and information-gain methods with benchmarks and clear approximation controls.

### Phase 9 — surrogate-model extensions

Add new model families when they expose a clean BoTorch-compatible posterior interface and provide capabilities not adequately covered by upstream models.
