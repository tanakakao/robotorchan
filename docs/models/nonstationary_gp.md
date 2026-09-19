# Phase 13: nonstationary Gaussian-process model

## Purpose

The robust-surrogate audit now covers sparse outliers, heavy tails,
heteroskedastic observation noise, replicate-derived noise, contamination, and
uncertain training inputs. A remaining distinct modeling gap is nonstationary
process behavior: the response smoothness itself can change across the design
space.

A stationary RBF or Matérn kernel uses one global lengthscale pattern. That can
be inadequate when, for example, a manufacturing process has a smooth operating
region and a sharp transition region.

Phase 13 evaluates a nonstationary kernel without conflating process
nonstationarity with observation-noise heteroskedasticity.

## Initial model

Use an input-dependent positive lengthscale field:

```text
ell(x) = softplus(a + B phi(x)) + ell_floor
```

with a low-complexity feature map `phi(x)`. For the first implementation,
`phi(x) = x` is sufficient.

A positive-semidefinite Gibbs kernel is appropriate:

```text
k(x, x')
= product_j [
    sqrt(2 ell_j(x) ell_j(x') / (ell_j(x)^2 + ell_j(x')^2))
  ]
  * exp(
      - sum_j (x_j - x'_j)^2
        / (ell_j(x)^2 + ell_j(x')^2)
    )
```

This allows local smoothness to vary while retaining a valid covariance
function.

## Statistical distinction

- heteroskedastic GP: observation variance changes with `x`;
- nonstationary GP: latent process covariance/smoothness changes with `x`;
- uncertain-input GP: training coordinates are uncertain;
- robust risk/scenario layers: decision evaluation changes, not the surrogate
  process itself.

These must remain separate concerns.

## Proposed API

```python
model = NonstationarySingleTaskGP(
    train_X,
    train_Y,
    lengthscale_floor=1e-3,
)
```

The model should retain the normal exact-GP contract:

- `supports_mll = True`;
- `make_mll()`;
- BoTorch posterior and acquisition compatibility;
- raw training data retention.

The local lengthscale parameters are learned jointly with the exact GP
hyperparameters.

## Complexity control

The first implementation should use an affine local-lengthscale field rather
than a neural network or second GP. This limits parameter count and keeps exact
GP training practical.

Do not add a second latent GP for lengthscales in Phase 13. Such a model changes
inference substantially and should be evaluated separately only if the affine
field is insufficient.

## Diagnostics

Expose:

```python
model.local_lengthscale(X)
```

returning positive local lengthscales with shape compatible with
`X[..., d]`.

This diagnostic is useful for identifying regions where the fitted process
changes rapidly.

## Required tests

1. local lengthscales are positive;
2. kernel covariance is symmetric;
3. kernel matrix is positive semidefinite up to numerical tolerance;
4. zero slope in the lengthscale field reduces to a stationary Gibbs/RBF-like
   covariance;
5. changing the local-lengthscale slope changes covariance;
6. gradients reach local-lengthscale parameters;
7. finite exact MLL;
8. raw training data, dtype/device migration, and state-dict roundtrip work;
9. posterior batch/q shapes work;
10. qMC acquisition smoke test passes;
11. existing heteroskedastic and robust models remain unchanged.

## Scope boundaries

Phase 13 does not add:

- neural or GP-driven lengthscale fields;
- input-dependent signal variance;
- change-point kernels;
- mixed/reduced/multitask cross-product wrappers;
- nonstationary observation noise.

Those require separate statistical justification.

## Exit criterion

Phase 13 is complete when robotorchan can represent input-dependent latent
smoothness through a documented positive-semidefinite covariance model while
preserving the standard exact-GP/BoTorch contract.
