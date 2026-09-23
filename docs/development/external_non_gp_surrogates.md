# External non-GP surrogate extension design

## Scope

This document defines the integration boundary for NGBoost, XGBoost, LightGBM, and CatBoost.
Phase 15 does not make these libraries core dependencies and does not expose placeholder public
surrogate classes. Adapters should be added only when their predictive semantics are implemented and
tested against the BoTorch posterior contract.

## Common rules

Every adapter must preserve raw `train_X` / `train_Y`, expose explicit `fit()`, and report
`supports_mll = False`. External estimators must not be presented as Gaussian Processes.

The adapter owns conversion from torch tensors to the backend representation. Candidate prediction
must return to the original torch dtype/device. CPU/numpy backends are treated as
`supports_input_gradients = False` unless a future native differentiable path is demonstrated.

Optional dependencies must be imported lazily or behind guarded imports. Installing robotorchan
without an external backend must continue to import successfully. Missing dependencies should fail
only when the corresponding adapter is constructed or used, with an actionable installation message.

No compatibility aliases, deprecated wrappers, or backend-specific public API forks should be added.

## Posterior strategies

### Complete-model bootstrap ensemble

XGBoost, LightGBM, and CatBoost can share the existing complete-model bootstrap concept. A posterior
member is one complete estimator fitted on one bootstrap resample. Individual boosting stages are
never posterior samples.

The shared implementation should generalize the current internal bootstrap infrastructure rather
than copy fitting and posterior-shape logic into each adapter.

### Native distributional prediction

NGBoost is distributional and should use its fitted predictive distribution rather than wrapping
individual boosting stages. Its adapter needs a posterior capable of sampling the backend predictive
distribution while preserving BoTorch `... x sample x q x m` semantics.

A Gaussian NGBoost distribution does not justify declaring all NGBoost configurations Gaussian:
the selected NGBoost distribution determines the predictive law. The adapter must inspect and
represent the configured distribution truthfully.

### Quantile prediction

Native quantile outputs from XGBoost, LightGBM, or CatBoost are not posterior samples by themselves.
A set of quantiles must not be converted silently into an `EnsemblePosterior` or a Gaussian
posterior. Quantile-specific acquisition support is deferred until a sampling/interpolation contract
is deliberately defined.

## Backend matrix

| Backend | Initial posterior route | Mixed/categorical route | Multi-output route | Dependency |
| --- | --- | --- | --- | --- |
| NGBoost | native predictive distribution | numeric first | explicit audit required | `ngboost` |
| XGBoost | complete-model bootstrap | encoded numeric first | explicit audit required | `xgboost` |
| LightGBM | complete-model bootstrap | native categorical where stable | explicit audit required | `lightgbm` |
| CatBoost | complete-model bootstrap | native categorical preferred | explicit audit required | `catboost` |

Backend version support must be pinned only after implementation tests establish the minimum working
version. Phase 15 intentionally avoids speculative version bounds.

## Mixed variables

The public robotorchan convention remains raw input plus `cat_dims`. Backend-specific category
representation is internal.

CatBoost is the preferred native-categorical target because categorical features are part of its
model semantics. LightGBM native categorical handling is also a candidate. XGBoost support should
only claim native categorical semantics after verifying the exact backend/version path; otherwise
the adapter must own an explicit encoder and document that choice.

Category columns must not overlap structural task, fidelity, context, or hierarchy dimensions.
Negative `cat_dims` should be normalized through the same robotorchan validation contract used by
existing mixed models.

## Acquisition optimization

The initial external adapters are gradient-free. They should work with MC acquisition functions when
their posterior is sampleable and use random/candidate/tree-style acquisition search.

Analytic Gaussian acquisition compatibility must not be inferred from a mean and variance API.
Gradient optimization must not be enabled merely because the external library exposes derivatives
with respect to model parameters.

## Multi-output and constraints

Multi-output support is backend-specific. An adapter may use independent output estimators when that
is the truthful backend contract, but member/sample alignment must be preserved across outputs.
Native joint-output behavior should be represented separately when available.

Constraints remain an acquisition/objective concern. External adapters model outputs; they do not
embed a feasibility decision rule.

## Proposed implementation order

1. Generalize internal complete-model bootstrap infrastructure so estimator construction and tensor
   conversion can be reused without exposing backend-specific compatibility layers.
2. Implement CatBoost first for the strongest native mixed/categorical value.
3. Implement LightGBM and XGBoost complete-model bootstrap adapters.
4. Implement NGBoost with a distribution-aware posterior rather than forcing it into the empirical
   ensemble abstraction. The Phase 14 contract is defined in
   [`ngboost_surrogate_design.md`](ngboost_surrogate_design.md).
5. Add backend-specific multi-output tests only after each backend's current API is audited.
6. Add MC acquisition and gradient-free optimizer integration tests.
7. Add optional-dependency smoke tests that verify base robotorchan imports without any backend.

## Acceptance criteria

An external adapter is ready for public export only when tests cover raw-data retention, missing
dependency behavior, deterministic seeding where supported, posterior shape/dtype/device, sampling,
MC acquisition integration, unsupported observation-noise behavior, and mixed/multi-output semantics
that the adapter publicly claims.

Documentation must state whether uncertainty comes from a native predictive distribution, bootstrap
complete-model disagreement, or another mechanism. These mechanisms must not be described
interchangeably.


## Phase 14 NGBoost decision

NGBoost is the next external probabilistic surrogate target. The first public adapter is intentionally
Gaussian-regression-only and must remain an optional dependency. Predictive variance / sample-based
regression AL is in scope; BALD is explicitly out of scope until an epistemic / aleatoric decomposition
is implemented and validated.
