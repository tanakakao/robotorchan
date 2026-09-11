# Architecture

## Core policy

robotorchan is a **BoTorch extension library**, not a Bayesian-optimization framework that wraps or replaces BoTorch.

Public APIs should therefore use native objects such as `torch.Tensor`, `botorch.models.model.Model`, `AcquisitionFunction`, `PosteriorTransform`, and BoTorch optimizers directly whenever practical.

## Clean-room development

Implementation in this repository is written from scratch from public papers, public documentation, and public upstream APIs. Source code, tests, internal documentation, naming conventions, or implementation details from prior private/internal projects must not be copied into robotorchan.

When an algorithm is derived from a paper, its implementation should document the reference and any deliberate deviations from the published method.

## Package layout

- `robotorchan.acquisition`: acquisition functions and active-learning criteria.
- `robotorchan.models`: surrogate models that expose BoTorch-compatible `Model` behavior.
- `robotorchan.objectives`: objectives, posterior transforms, and constraint-related helpers when BoTorch does not already provide them.
- `robotorchan.optim`: acquisition optimization and search-space utilities that extend, rather than duplicate, `botorch.optim`.

Additional top-level namespaces should only be introduced when a stable group of functionality exists.

## API rules

1. Prefer subclassing or composing BoTorch abstractions over introducing parallel abstractions.
2. Keep algorithm configuration explicit in constructors.
3. Preserve tensor batch semantics used by BoTorch.
4. Avoid application-specific configuration objects in the core package.
5. Do not hide model fitting or acquisition optimization behind implicit global state.
6. New public APIs require tests for shapes, dtype/device behavior, gradients when applicable, and numerical edge cases.
7. Experimental APIs may live under normal modules, but must be clearly documented as unstable until promoted.

## Roadmap

### Phase 1 — foundation

- package metadata and `src/` layout;
- linting, tests, and CI;
- architecture and contribution conventions.

### Phase 2 — boundary and uncertainty acquisition functions

Start with small, well-tested analytic criteria useful for active learning and level-set / boundary search, such as Straddle-style acquisition functions. Use BoTorch posterior transforms for multi-output scalarization rather than reinventing scalarization.

### Phase 3 — classification active learning

Add predictive entropy, probability variance, margin-based uncertainty, BALD-style criteria, and boundary-focused methods behind BoTorch-compatible acquisition-function interfaces where the model posterior semantics permit it.

### Phase 4 — multi-objective and constrained extensions

Add methods that are not already available upstream, emphasizing composability with BoTorch objectives, constraints, partitioning, and MC samplers.

### Phase 5 — robust and risk-aware optimization

Add input-uncertainty and risk criteria such as variance / CVaR-oriented extensions when they provide functionality beyond BoTorch's existing risk-measure APIs.

### Phase 6 — ordinal and preference optimization

Add ordinal-target and preference-aware acquisition functions with explicit assumptions and dedicated tests.

### Phase 7 — lookahead and information-theoretic methods

Add computationally heavier multi-step, entropy, and information-gain methods with benchmarks and clear approximation controls.

### Phase 8 — surrogate-model extensions

Only add model families when they expose a clean BoTorch-compatible posterior interface and provide capabilities not adequately covered by upstream models.
