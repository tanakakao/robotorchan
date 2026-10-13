# API unification audit — Phase 1 inventory and decisions

Status: Phase 1 baseline (source: `main`, inspected 2026-10-13).
Scope: public surrogate-model and acquisition APIs, including the convenience `fit()` request.
This phase is **inventory and planning**: it does not assert all model signatures or runtime
capabilities have been tested. Phases 2–3 must confirm per-class details before changes.

## Source of truth

| Concern | Source |
| --- | --- |
| Public regression/model exports | `src/robotorchan/models/__init__.py` (`__all__`) |
| Model capability registry | `src/robotorchan/models/registry.py` |
| Classification model IDs | `src/robotorchan/models/classification/registry.py` and `binary/registry.py` |
| Shared training contract | `src/robotorchan/models/base.py` |
| Public acquisition exports | `src/robotorchan/acquisition/__init__.py` |
| Acquisition capability registry | `src/robotorchan/acquisition/registry.py` |
| Semantic composition | `src/robotorchan/acquisition/composition.py` |
| Public API tests | `tests/models/contracts/`, `tests/acquisition/` |
| Existing coverage artifact | `scripts/generate_model_coverage.py`, `docs/model_coverage.json` |
| Existing training guide | `docs/development/training_api.md` |

The inspected `models.__all__` contains 154 exported names and
`acquisition.__all__` contains 44 exported names. These are **public symbol counts**,
**not class or implementation counts**; they include metadata, helpers and exceptions.
Do not use these totals as claims about the number of models or acquisitions.

## First-pass training API inventory

| Family | Existing observable contract | Follow-up |
| --- | --- | --- |
| Exact GP | `ExactGPModelMixin.make_mll()` returns an `ExactMarginalLogLikelihood` | Check all inherited models and provide a convenient training entry point without replacing BoTorch internals |
| Variational GP | `SingleTaskVariationalGP.make_mll(num_data=None)` | Verify ELBO sizing, optimizer flow and minibatch behavior |
| Robust Student-t / contaminated / joint heteroskedastic | Explicit `training_loss()`; some `make_mll()` methods intentionally reject | Keep model-specific loss; a training facade must not silently run an exact MLL |
| Non-GP regressors | `FittableModelMixin.fit()`, `supports_fit=True` in mixin; `RandomForestSurrogate.fit()` fits stored data | Preserve no-argument convenience call; verify other estimators individually |
| Binary classification | Dedicated registry and predictive probability/latent distinction | Determine fit and loss requirements per subfamily |
| Deep and learned representations | Several classes expose `training_loss()` | Audit joint optimization and alternating training, not only method names |
| HeterogeneousModel | Child-model container without a synthetic shared posterior | Training must respect each child's inference and not invent a joint GP |
| Fully Bayesian / MCMC | To be verified | Distinguish MCMC fitting from ordinary MLL and avoid misleading `fit()` behavior |

Notable verified examples:
- `RandomForestSurrogate.fit()` fits from constructor-provided raw tensors and
  `posterior()` requires the model to have been fitted.
- `ModelTrainingMixin` defines a `supports_fit=False` default; the
  `FittableModelMixin` provides an explicit abstract `fit()` with
  `supports_fit=True`.
- `HeterogeneousModel` explicitly lacks a shared `posterior()` by design.

## Intended fit() experience (decision proposal for Phase 4)

**Goal:** `model.fit()` is a discoverable, practical convenience where well-defined.
It is not a requirement for unrelated families to have identical fitting algorithms.

- A zero-argument `fit()` should fit the constructor's training data by default.
- Exact/variational fitting may use the correct native BoTorch/GPyTorch training
  routine internally; model-specific training objectives remain authoritative.
- Configurable optimization belongs to a documented, small set of keyword
  arguments or a well-defined configuration object, with no implicit `**kwargs`
  swallowing and no fictitious universal `make_mll()`.
- `torch.nn.Module.train()` and `.eval()` are **mode switches**; do not conflate
  these with `fit()`.
- Where generic fitting is not safely defined, expose an explicit unsupported
  capability and an actionable error instead of an incorrect approximation.
- Never silently return a fitted model with stale data, move between CPU/GPU
  unexpectedly, or change the semantics of `raw_train_X/Y`.
- Decide return type (`None` or `self`), optional progress/history,
  reproducibility and refitting behavior **before** implementing widely.
- Preserve native BoTorch low-level interfaces (`posterior`, `make_mll`,
  external optimizer, samplers and acquisition constructors).

## Acquisition inventory boundary

Local public API exports include active-learning criteria (`BALD`,
`PredictiveEntropy`, `ProbabilityVariance`, `MarginUncertainty`,
`Straddle`, `PosteriorVariance`, etc.) plus semantic composition helpers
(`make_qei_acquisition`, `make_qnei_acquisition`, `make_qehvi_acquisition`,
`make_qnehvi_acquisition`, `make_qlognparego_acquisition` and constraint bridges).

Registry metadata also covers upstream BoTorch acquisitions that must **not**
be duplicated as robotorchan classes merely for namespace symmetry.
`docs/development/acquisition-architecture.md` is the governing policy.

Audit per acquisition:
- origin (native BoTorch / custom / composition helper);
- constructor arguments and whether `objective`, `posterior_transform`,
  `sampler`, `X_pending`, `constraints` actually apply;
- `forward(X)` contract, q-batch, output shapes, devices/dtypes;
- posterior requirement, observation semantics, sample vs probability
  constraints, multi-output, one-shot and fantasize;
- compatible optimizers, including candidate vs outcome constraints;
- runtime tests, documentation and truthful registry capabilities.

## Coverage categories for Phases 2–3

Classification: implemented-and-exposed / implemented-but-not-exposed /
metadata-only / missing / intentionally-unsupported.
Record evidence (symbol, source, test), severity and next action for every gap.

Model families: standard exact, multitask/multioutput, mixed variables,
multi-fidelity, fully Bayesian, high-dimensional/reduced, expressive/deep,
robust/uncertain, non-GP, ensemble, binary classification, heterogeneous.

Acquisition families: analytic and MC improvement, log-improvement,
multiobjective, constrained BO, lookahead, information-theoretic,
multi-fidelity, posterior sampling, regression AL, classification AL,
semantic composition.

## Fixed follow-up work and acceptance gates

**Phase 2: Model API audit**
1. Extract per-class constructor and method signatures via `inspect.signature`.
2. Include inherited `fit`, `make_mll`, `training_loss`, `posterior`,
   `condition_on_observations`, `fantasize`, classification prediction and
   raw-data provenance.
3. Compare advertised capability with actual callable behavior and tests.
4. Mark `fit()` cases: present / safely implementable / special training /
   intentionally unavailable.

**Phase 3: Acquisition API audit**
1. Map each public acquisition and selected native BoTorch registry name.
2. Verify constructor, batch contract, posterior needs and optimization routes.
3. Identify implemented/unconnected vs genuinely missing functionality.

**Phase 4+: implementation only after gap classification.**
Keep no monkey-patches, legacy aliases or duplicate native wrappers.
Avoid breaking imports, registry identifiers, existing notebook flows and
classification latent/probability semantics.

### Phase 1 gate

- [x] Identify authoritative public exports, registries and documentation.
- [x] Establish a first-pass training/fitting and acquisition inventory.
- [x] Explicitly include a convenient `fit()` in the work scope.
- [x] Record non-negotiable BoTorch and heterogeneous semantics.
- [ ] Generate exhaustive per-class signature/capability matrix (Phase 2).
- [ ] Verify all method contracts by executing tests (Phases 2–3, then 10).

No runtime test or exhaustive per-class audit is claimed by this document.
