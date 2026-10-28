# Mixed Gaussian uncertain-input GP design

## Purpose

This document defines the probability model for combining Gaussian uncertainty in
continuous observed training inputs with deterministic categorical design
variables. The model addresses uncertainty in the recorded continuous location of a
training observation. It does not model uncertain category membership.

## Probability model

Partition each raw design row as

```text
x = (x_cont, c)
```

where `x_cont` contains ordinary continuous design variables and `c` contains
deterministic nominal categorical variables. The latent continuous training
location is

```text
X_cont* ~ Normal(mu_cont, Sigma_cont)
```

while `c` is observed exactly.

For two observations `i` and `j`, use

```text
k((mu_i, Sigma_i, c_i), (mu_j, Sigma_j, c_j))
  = E[k_cont(X_i*, X_j*)] * k_cat(c_i, c_j)
```

where the expected continuous covariance is the same analytic Gaussian
marginalization used by `GaussianUncertainInputKernel`. The categorical
covariance is native and never receives Gaussian perturbations.

This product form is deliberately narrower than the additive-plus-interaction
kernel used by ordinary `MixedSingleTaskGP`: adding a categorical-only term
would encode a separate category effect that is not part of this uncertain-input
probability model. Such an extension requires a separate statistical decision.

## Input contract

The public model should accept raw mixed-space `train_X`, `train_Y`, and
`cat_dims`. Input uncertainty is specified only for the continuous dimensions:

- `train_X_std`: shape `(..., n, d_cont)`; or
- `train_X_covar`: shape `(..., n, d_cont, d_cont)`.

Exactly one representation is required. The order of uncertainty coordinates
follows increasing raw continuous feature index after normalizing `cat_dims`.

The model must reject:

- uncertainty tensors whose width includes categorical dimensions;
- non-finite or negative standard deviations;
- non-finite, asymmetric, or non-positive-semidefinite covariance matrices;
- all-categorical inputs, because there is no Gaussian-uncertain continuous
  coordinate to marginalize;
- duplicate or out-of-range categorical dimensions.

No silent reshaping, renormalization, category jitter, or ordinal interpretation
of category codes is allowed.

## Kernel representation

The expected continuous kernel needs an augmented representation containing the
continuous mean and flattened continuous covariance. The categorical columns
remain deterministic.

A dedicated private kernel should split this augmented representation into:

1. continuous mean/covariance fields for the expected RBF calculation; and
2. deterministic categorical fields for a native `CategoricalKernel`.

The resulting covariance is their product. Reuse validation and expected-RBF
math from the existing uncertain-input implementation rather than maintaining a
second formula.

The implementation should factor the continuous uncertainty validation and
augmentation into private helpers before adding the mixed public class.

## Candidate posterior semantics

Optimization candidates are deterministic raw mixed-space points. Their
continuous covariance is zero, and their category values are passed unchanged
to the categorical covariance.

Therefore `posterior(X)` accepts the same raw mixed tensor shape expected by
other mixed models. Acquisition functions can evaluate the model without an
external augmentation transform.

Candidate-side perturbation robustness remains a separate uncertainty/scenario
layer and must not be folded into this training-input model.

## API

```python
MixedUncertainInputSingleTaskGP(
    train_X,
    train_Y,
    *,
    cat_dims=[1, 3],
    train_X_std=continuous_std,
)
```

or

```python
MixedUncertainInputSingleTaskGP(
    train_X,
    train_Y,
    *,
    cat_dims=[1, 3],
    train_X_covar=continuous_covar,
)
```

The model is an exact GP and follows the normal robotorchan exact-model
contract:

- `supports_mll=True`;
- `make_mll()` returns an exact marginal log likelihood;
- raw mixed `train_X`, `train_Y`, uncertainty tensors, and normalized
  `cat_dims` are retained as provenance;
- no compatibility alias is added.

## Test requirements

Implementation must cover:

- diagonal and full-covariance continuous uncertainty;
- one and multiple deterministic categorical dimensions;
- exact one-hot-like category identity behavior through native categorical
  covariance, without ordinal distances;
- kernel symmetry and positive-semidefinite covariance;
- zero continuous uncertainty reducing to deterministic continuous RBF times
  categorical covariance;
- posterior and qUCB on raw mixed candidates;
- invalid uncertainty shapes and all-categorical rejection;
- dtype/device migration and state-dict round trip;
- public-model and mixed-model inventory contracts.

## Non-goals

This phase does not support probabilistic category membership. That remains the
separate `UncertainCategoricalSingleTaskGP` concern. It also does not create a
cross-product with reduced, multitask, ALEBO, or robust-risk classes.

## Implementation decision

Retain this model for implementation. The probability model is distinct and
well-defined: Gaussian uncertainty affects only continuous observed training
locations, while deterministic nominal categories enter through native
categorical covariance. It cannot be represented correctly by passing integer
category codes through `UncertainInputSingleTaskGP`.
