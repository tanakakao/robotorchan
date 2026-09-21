# Non-GP probabilistic surrogate audit

## Scope

This audit defines the integration boundary for tree ensembles, boosting models,
Bayesian neural networks (BNNs), and related probabilistic surrogates. It is a
design gate: no public surrogate class is added in this phase.

Current upstream BoTorch semantics and the current `main` branch are
authoritative.

## Findings

### BoTorch posterior boundary

BoTorch's model interface is already suitable for non-GP surrogates. Monte
Carlo acquisition functions require a model `posterior(X)` that returns a
`Posterior` capable of sampling. Gaussianity is not a general requirement.

`EnsemblePosterior` is therefore the preferred first integration path for
surrogates that naturally expose predictive function samples. Its ensemble
values follow the BoTorch convention

```text
... x s x q x m
```

where `s` is the ensemble/sample dimension.

This covers two important families without inventing a Gaussian posterior:

- tree ensembles: one prediction trajectory per tree or fitted ensemble member;
- neural ensembles / sampled BNN predictions: one trajectory per network or
  posterior weight sample.

A dedicated robotorchan posterior should only be introduced when the upstream
posterior cannot preserve the required semantics.

### Existing robotorchan contracts

`SupervisedTrainingDataMixin` already provides the correct provenance contract
for supervised non-GP models through `raw_train_X`, `raw_train_Y`, and
`raw_train_Yvar`.

`ModelTrainingMixin` already distinguishes MLL-capable and non-MLL models with
`supports_mll`. Non-GP models must not fake a marginal likelihood merely to
satisfy a common API. Their fitting procedure belongs in an explicit training
contract added in a later phase.

The existing `RandomSearchStrategy` is immediately relevant to tree models.
Tree predictions are not differentiable with respect to candidate inputs, so
gradient-free acquisition search is required. Search strategy remains separate
from surrogate semantics.

### Existing BNN-named model

`InfiniteWidthBNNGP` is an exact GP with an infinite-width neural-network
kernel. It is not a finite Bayesian neural network and must remain in the GP
family.

A future finite-width BNN must therefore use a distinct public name and must not
replace, alias, or wrap `InfiniteWidthBNNGP` for compatibility.

### Dependency boundary

The core package currently depends only on BoTorch. scikit-learn is not a core
dependency.

Tree and sklearn boosting support therefore requires an explicit dependency
decision before implementation. The preferred direction is an optional
dependency group unless later benchmarking demonstrates that tree surrogates
should be part of the minimal installation.

A PyTorch-native BNN does not inherently require scikit-learn. Additional
inference libraries such as Pyro must also remain optional unless the selected
BNN inference method requires them.

## Surrogate taxonomy

The implementation should use statistical semantics rather than one class
hierarchy for every model:

```text
Probabilistic surrogate
├── GP posterior
│   └── existing robotorchan GP families
├── Empirical ensemble posterior
│   ├── Random Forest
│   ├── Extra Trees
│   ├── bootstrap boosting
│   └── deep ensemble
├── Bayesian neural posterior
│   └── finite-width BNN posterior samples
├── Distributional posterior
│   └── future NGBoost-like models
└── Quantile prediction
    └── gradient / histogram gradient boosting
```

Quantile predictions are not posterior samples by themselves. They must not be
silently converted into a Gaussian posterior.

## BNN design requirements

The finite-width BNN path must preserve the distinction between epistemic and
observation uncertainty where the inference method supports it.

Candidate approaches to evaluate before choosing the first implementation are:

1. variational weight posterior;
2. MCMC weight posterior;
3. Laplace approximation;
4. deep ensemble as a non-Bayesian empirical-posterior baseline.

Deep ensemble must not be named BNN. It shares an ensemble posterior mechanism
but has different statistical semantics.

For gradient-based acquisition optimization, BNN posterior samples must retain
autograd from candidate `X` through the sampled network prediction. Detaching
candidate inputs in the posterior path is forbidden for this route.

## Acquisition compatibility

Initial compatibility should be capability based.

| Surrogate path | MC acquisition | Analytic Gaussian acquisition | Gradient search |
| --- | --- | --- | --- |
| RF / Extra Trees | target | no general guarantee | no |
| bootstrap boosting | target | no general guarantee | no |
| finite BNN | target | no general guarantee | target |
| deep ensemble | target | no general guarantee | target |
| quantile boosting | deferred | no | model dependent |

The project should prefer MC acquisition functions for non-Gaussian posteriors.
Analytic acquisition support must only be claimed when its distributional
assumptions are actually satisfied.

## Mixed-space implications

Tree models are a strong mixed-space target, but sklearn regressors consume
numeric arrays rather than robotorchan's semantic raw mixed representation.
The public API should continue to accept raw inputs and `cat_dims`; any
encoding must be model-owned.

A finite BNN can reuse the same model-owned categorical encoding principle when
mixed support is added. Structural task, fidelity, context, and hierarchy
columns remain excluded from ordinary `cat_dims`.

## Recommended implementation sequence

Phase 2 should establish the smallest non-GP supervised training and posterior
contract without changing existing GP behavior.

Phase 3 should validate empirical ensemble posterior behavior independently of
sklearn using a small deterministic test ensemble. This keeps posterior-shape
and acquisition tests independent of optional dependencies.

Random Forest and Extra Trees should follow after that. Finite BNN should reuse
the posterior contract but remain a separate neural probabilistic family.

Boosting should be added only after ensemble semantics are stable because an
individual boosting tree is not an independent posterior member.

## Phase 1 decisions

The following decisions are closed by this audit:

- use BoTorch `Model` / `Posterior` semantics instead of a parallel BO API;
- prefer upstream `EnsemblePosterior` where its semantics fit;
- keep `make_mll()` truthful and unsupported for models without an MLL;
- keep surrogate fitting separate from acquisition search;
- use gradient-free search for sklearn tree ensembles;
- treat finite BNN as non-GP and keep `InfiniteWidthBNNGP` as a GP;
- design empirical ensemble support so tree ensembles and deep ensembles can
  share infrastructure without sharing statistical names;
- do not interpret quantile intervals as Bayesian posterior samples;
- keep new third-party ML dependencies optional until dependency policy is
  deliberately changed.

## Phase 2 gate

Before adding public RF, Extra Trees, or BNN classes, Phase 2 must define and
test:

- non-GP supervised fitting capability;
- `raw_train_X/Y` preservation;
- posterior shape and dtype/device behavior;
- explicit `supports_mll = False`;
- serialization expectations;
- capability reporting needed by acquisition/search integration.

No compatibility aliases or deprecated wrappers are to be introduced.
