# Ensemble MAP-SAAS × multi-fidelity audit

## Scope

Phase 38 audits whether an `EnsembleMapSaasMultiFidelityGP` can be expressed
without weakening either BoTorch's ensemble semantics or its native
multi-fidelity covariance semantics. This phase does not add a public model.

## Upstream contracts

Current BoTorch `EnsembleMapSaasSingleTaskGP` is not just a MAP-SAAS kernel
choice. It is an ensemble model with these observable contracts:

- `_is_ensemble = True`.
- The raw training set is repeated over a `num_taus` model batch.
- Each ensemble member owns a distinct SAAS `tau`.
- Mean, Matern covariance, scale, and likelihood use the ensemble batch shape.
- `posterior` inserts the ensemble dimension and returns a
  `GaussianMixturePosterior`, rather than advertising one collapsed Gaussian.

Current BoTorch `SingleTaskMultiFidelityGP` builds its fidelity covariance with
`_setup_multifidelity_covar_module`. With `linear_truncated=False`, the
non-fidelity covariance is multiplied by native `ExponentialDecayKernel`
and/or `DownsamplingKernel` terms. Those fidelity kernels are constructed with
the model's augmented batch shape.

## What is reusable

The covariance composition itself is compatible in principle. A design
covariance can be supplied to `SingleTaskMultiFidelityGP`, and its native
fidelity kernels can share an ensemble batch shape.

The required conceptual structure is:

```text
raw X
  |
  +-- design dims ------> batched MAP-SAAS Matern kernels, one tau per member
  |
  +-- data fidelity ----> batched native DownsamplingKernel
  |
  +-- iteration fidelity -> batched native ExponentialDecayKernel
  |
  +---------------------> member-wise exact multi-fidelity covariance
                              |
                              +--> Gaussian mixture posterior
```

Fidelity dimensions must never participate in SAAS design relevance.

## Public API seam assessment

Phase 36 found a public BoTorch factory for additive MAP-SAAS covariance. The
ensemble model does not expose an equivalent public factory that constructs its
batched design covariance independently of the complete model.

Its constructor currently creates the ensemble batch, samples or validates
`taus`, constructs a batched Matern kernel, attaches the SAAS prior, builds a
batched scale kernel and likelihood, and then defines mixture posterior
semantics at the model level.

Therefore simply passing the existing complete
`EnsembleMapSaasSingleTaskGP` into `SingleTaskMultiFidelityGP` is invalid:
a complete model is not a covariance module, and doing so would lose clear
ownership of likelihood, model batch, and posterior semantics.

Copying that constructor's private composition into a public robotorchan model
would also create avoidable maintenance coupling to BoTorch internals.

## Posterior and acquisition boundary

A future implementation must preserve `GaussianMixturePosterior`. It must not
collapse the ensemble to a Gaussian merely to reuse an acquisition function.

Consequently acquisition compatibility must be established from the actual
posterior contract. In particular, Phase 37's successful Gaussian
multi-fidelity KG validation for `AdditiveMapSaasMultiFidelityGP` is not
evidence that ensemble MAP-SAAS × MF supports the same KG path.

The following require independent runtime evidence:

1. posterior shape and finite mixture samples;
2. ensemble-aware MC acquisition behavior;
3. conditioning and fantasize lifecycle;
4. cost-aware multi-fidelity acquisition compatibility;
5. fitting with the ensemble model batch and exact MLL.

## Implementation gate

Do not add `EnsembleMapSaasMultiFidelityGP` yet.

Implementation is justified only when at least one stable route exists:

1. BoTorch exposes a public ensemble MAP-SAAS design-covariance factory or
   another public composition seam that preserves its ensemble contract; or
2. robotorchan deliberately owns a small ensemble MAP-SAAS covariance builder,
   with the statistical contract documented and tested independently of
   BoTorch's private constructor implementation.

Route 2 requires a separate design decision because it is no longer a thin
BoTorch wrapper.

## Required evidence before public export

A future implementation must prove all of the following:

- raw `train_X/train_Y/train_Yvar` retention;
- normalized negative fidelity indices;
- duplicate fidelity-role rejection;
- at least one design dimension remains;
- fidelity dimensions are absent from every SAAS member's design ARD space;
- one explicit `tau` per ensemble member and stable state-dict ownership;
- native fidelity kernels carry the same ensemble batch shape;
- posterior type remains `GaussianMixturePosterior`;
- posterior sampling is finite and preserves the ensemble dimension semantics;
- exact MLL fitting contract is valid;
- conditioning and fantasize are safe;
- supported acquisitions are determined by runtime evidence, not inherited
  from the Gaussian single-model MF variants.

## Phase 38 decision

`EnsembleMapSaasMultiFidelityGP` remains **research-gated**.

The mathematical covariance composition is plausible, but current public
BoTorch APIs do not provide the same clean public construction seam used by the
additive Phase 37 model. More importantly, the ensemble model's
`GaussianMixturePosterior` is part of its statistical contract and must be
preserved explicitly.

The next implementation phase should not create this public model unless the
project intentionally accepts ownership of the missing ensemble covariance
builder and validates mixture-posterior acquisition semantics.
