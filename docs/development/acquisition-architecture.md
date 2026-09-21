# Acquisition architecture

robotorchan follows a **BoTorch-first** acquisition design.

## Scope

The acquisition package contains only behavior that adds robotorchan-specific value.
BoTorch acquisition classes that already satisfy the required contract are imported and
constructed directly from BoTorch. robotorchan does not create aliases, compatibility
wrappers, or duplicated implementations merely to provide a unified namespace.

Current robotorchan-owned acquisition code is limited to compatibility validation for
non-GP empirical ensemble surrogates. Future custom implementations may be added for
regression active learning, level-set estimation, and methods that are not available in
BoTorch.

## Ownership rules

An acquisition belongs in `robotorchan.acquisition` only when at least one condition holds:

- robotorchan adds model-specific validation that BoTorch does not provide;
- the method itself is not provided by the supported BoTorch version;
- robotorchan needs additional semantics that cannot be expressed by normal BoTorch
  constructor arguments.

Otherwise users should instantiate the BoTorch acquisition directly.

## Non-GP surrogates

Tree ensemble surrogates expose empirical posteriors. They therefore use BoTorch Monte
Carlo acquisition functions. Analytic Gaussian acquisition functions are rejected by
`validate_non_gp_acquisition`.

`make_non_gp_acquisition` remains a small construction helper because it combines
construction with the robotorchan-specific compatibility check. It must not grow into a
general acquisition factory.

## Deferred classification support

Classification models and classification-specific acquisition functions are intentionally
outside the current scope. Predictive entropy, BALD for classification, margin uncertainty,
least confidence, probability variance, and related criteria will be designed when
classification surrogate models are introduced.

Regression level-set estimation is not classification and remains in scope for future
phases.

## Planned package shape

Only robotorchan-owned implementations should create modules:

```text
acquisition/
├── __init__.py
├── non_gp.py
├── active_learning/
│   ├── variance.py
│   ├── straddle.py
│   ├── boundary.py
│   └── epig.py
└── sampling/
    └── thompson.py
```

Directories are created when their first implementation lands. Empty scaffolding is avoided.

## Public API contract

- BoTorch-native acquisitions are not re-exported from robotorchan.
- No acquisition registry is introduced unless dispatch requirements emerge from concrete
  implementations.
- No string-based acquisition factory is introduced in advance.
- No legacy aliases or deprecated wrappers are added.
- Custom acquisitions must follow BoTorch's `AcquisitionFunction` conventions whenever
  practical so that BoTorch optimizers can consume them directly.
- Model capability checks should fail explicitly and close to construction/evaluation.

This keeps acquisition functions composable with BoTorch while avoiding a second API layer
that would need to track BoTorch releases.
