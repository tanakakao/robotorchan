# Phase 4: Nonstationary and spectral-mixture GP classification

This guide covers **multiclass** (unordered categories) and **ordinal**
(ordered categories) classification with nonstationary Gibbs kernels and
spectral-mixture kernels. All models use existing BoTorch/GPyTorch variational
GP training and expose `fit()`, `predict_proba()`, and `posterior()`.

## When to choose each kernel

- **Gibbs nonstationary**: the latent function changes smoothness across
  input locations. Its local lengthscale is a learned function of the input.
- **Spectral mixture**: the latent function exhibits periodic or
  multi-frequency structure; the stationary kernel models a mixture of
  spectral components.
- **Mixed variants**: use the existing mixed covariance construction to
  combine continuous and categorical inputs. The categorical columns must
  be integer-coded and listed in `cat_dims`.
- **Multiclass**: one latent GP per class and a categorical softmax model.
- **Ordinal**: one latent GP and an ordered-probit likelihood with
  ordered thresholds. Do not interpret ordinal class indices as continuous
  regression targets.

A nonstationary Gibbs kernel and a spectral-mixture kernel address
*different* inductive biases. A spectral mixture is not generally a
nonstationary covariance.

## Minimal candidate proposal

```python
import torch

from robotorchan.models.classification import (
    NonstationaryMulticlassGPClassifier,
    SpectralMixtureOrdinalGPClassifier,
)

torch.manual_seed(0)
X = torch.linspace(-1, 1, 24, dtype=torch.double).unsqueeze(-1)
Y = torch.bucketize(X.squeeze(-1), torch.tensor([-0.3, 0.3], dtype=X.dtype))
candidates = torch.linspace(-1, 1, 101, dtype=X.dtype).unsqueeze(-1)

for model_type in (
    NonstationaryMulticlassGPClassifier,
    SpectralMixtureOrdinalGPClassifier,
):
    model = model_type(X, Y, num_classes=3)
    model.fit(epochs=20, lr=0.01)
    probabilities = model.predict_proba(candidates)
    entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
    next_x = candidates[entropy.argmax()]
    print(model_type.__name__, next_x, probabilities.shape)
```

For mixed inputs, concatenate categorical columns to the feature tensor
and use `MixedNonstationaryMulticlassGPClassifier`,
`MixedNonstationaryOrdinalGPClassifier`,
`MixedSpectralMixtureMulticlassGPClassifier`, or
`MixedSpectralMixtureOrdinalGPClassifier`, passing `cat_dims=[...]`.
Candidate sets must contain valid categorical codes.

## Mathematical basis

A Gibbs covariance with input-dependent lengthscales in one dimension is

```math
k(x,x') = \sigma^2 \sqrt{\frac{2\ell(x)\ell(x')}{\ell(x)^2+\ell(x')^2}}
\exp\left(-\frac{(x-x')^2}{\ell(x)^2+\ell(x')^2}\right).
```

The spectral-mixture kernel is derived from a Gaussian mixture in the
spectral domain (Bochner's theorem). For a stationary covariance,

```math
k(\tau) = \sum_{q=1}^Q w_q
\exp(-2\pi^2 \tau^\top \Sigma_q\tau)
\cos(2\pi\mu_q^\top\tau).
```

The multiclass predictive probabilities integrate a softmax over the
latent GP posterior; the ordinal model integrates ordered-probit
probabilities over its single latent GP. These are not equivalent
likelihoods.

## References

- Gibbs (1997), *Bayesian Gaussian Processes for Regression and Classification*.
- Wilson and Adams (2013), *Gaussian Process Kernels for Pattern Discovery
  and Extrapolation*, ICML.
- Rasmussen and Williams (2006), *Gaussian Processes for Machine Learning*.
- GPyTorch documentation: GibbsKernel and SpectralMixtureKernel.
