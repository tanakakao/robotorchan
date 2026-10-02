# Acquisition architecture

robotorchan follows a **BoTorch-first** acquisition design.

## Scope

The acquisition package contains only behavior that adds robotorchan-specific value.
BoTorch acquisition classes that already satisfy the required contract are imported and
constructed directly from BoTorch. robotorchan does not create aliases, compatibility
wrappers, or duplicated implementations merely to provide a unified namespace.

Current robotorchan-owned acquisition code covers:

- regression active-learning and level-set criteria that are not provided directly by BoTorch;
- compatibility checks between registered model and acquisition capabilities;
- acquisition / optimizer semantic compatibility checks;
- posterior-sampler selection from registered model capabilities;
- non-GP Monte Carlo acquisition validation;
- finite-set Thompson-style candidate selection.

The acquisition registry is metadata for workflow recommendation and compatibility dispatch.
It does not replace BoTorch constructors and is not a string-based acquisition factory.

## Ownership rules

An acquisition implementation belongs in `robotorchan.acquisition` only when at least one
condition holds:

- robotorchan adds model-specific validation that BoTorch does not provide;
- the method itself is not provided by the supported BoTorch version;
- robotorchan needs additional semantics that cannot be expressed by normal BoTorch
  constructor arguments.

Otherwise users should instantiate the BoTorch acquisition directly. BoTorch-native
acquisitions may still have registry metadata when workflow dispatch needs to reason about
their capabilities.

## Active learning

`active_learning/` owns robotorchan acquisition implementations for regression active
learning and level-set estimation. These classes follow BoTorch acquisition conventions so
that normal BoTorch optimizers can consume them when their semantics permit.

Classification-specific criteria remain deferred until classification surrogate models are
introduced.

## Capability and compatibility metadata

`capabilities.py` defines acquisition metadata such as posterior requirements, batch
semantics, multi-output support, fantasization requirements, and one-shot behavior.

`registry.py` records both robotorchan-owned acquisitions and selected BoTorch-native
acquisitions needed by workflow recommendation or compatibility dispatch. Registry entries
describe capabilities; they do not wrap or reconstruct the BoTorch implementation.

`compatibility.py` checks model / acquisition compatibility using model and acquisition
registry metadata. `optimizer_compatibility.py` separately checks whether an optimization
strategy can preserve acquisition semantics such as one-shot augmented batches.

Candidate/input-space feasibility constraints remain an optimizer responsibility and must
not be inferred from acquisition output-constraint metadata.

## Sampling responsibilities

Sampling has two distinct responsibilities and they intentionally remain separate:

- `samplers.py` selects a native BoTorch `MCSampler` from registered posterior-sampling
  capabilities. It is capability dispatch, not a candidate-generation algorithm.
- `sampling/` owns posterior-sampling candidate-selection algorithms. For example,
  `sampling/thompson.py` uses BoTorch `MaxPosteriorSampling` to select candidates from a
  finite set.

Do not merge these responsibilities merely because both involve posterior samples.

## Non-GP surrogates

Non-GP surrogates use BoTorch Monte Carlo acquisition functions when their posterior
contract supports sampling. `validate_non_gp_acquisition` validates the acquisition and
sampler against registered posterior-sampling capabilities rather than assuming every
non-GP model is an empirical ensemble.

`make_non_gp_acquisition` remains a small construction helper because it combines
construction with robotorchan-specific compatibility validation. It must not grow into a
general acquisition factory.

## Deferred classification support

Classification models and classification-specific acquisition functions are intentionally
outside the current scope. Predictive entropy, BALD for classification, margin uncertainty,
least confidence, probability variance, and related criteria will be designed when
classification surrogate models are introduced.

Regression level-set estimation is not classification and is represented by the current
active-learning package.

## Current package shape

Only robotorchan-owned implementations or concrete cross-layer metadata / dispatch
responsibilities create modules:

```text
acquisition/
├── __init__.py
├── capabilities.py
├── compatibility.py
├── non_gp.py
├── optimizer_compatibility.py
├── registry.py
├── samplers.py
├── active_learning/
│   ├── epig.py
│   ├── randomized_straddle.py
│   ├── straddle.py
│   └── variance.py
└── sampling/
    └── thompson.py
```

Empty scaffolding is avoided.

## Public API contract

- BoTorch-native acquisitions are not re-exported from robotorchan.
- The acquisition registry is capability metadata, not a string-based factory.
- Registry entries for BoTorch-native acquisitions must not create robotorchan wrapper
  classes solely for namespace uniformity.
- No legacy aliases or deprecated wrappers are added.
- Custom acquisitions follow BoTorch's `AcquisitionFunction` conventions whenever
  practical so that BoTorch optimizers can consume them directly.
- Model capability checks fail explicitly and close to construction, recommendation, or
  evaluation.
- Candidate feasibility remains owned by `robotorchan.optim`; acquisition metadata only
  describes acquisition semantics.

This keeps acquisition functions composable with BoTorch while allowing workflow and
optimization layers to reason about concrete compatibility requirements without building a
second acquisition API.
