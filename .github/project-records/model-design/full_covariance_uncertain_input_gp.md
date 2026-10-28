# Full-covariance uncertain training inputs

## Purpose

`UncertainInputSingleTaskGP` supports diagonal Gaussian uncertainty and **correlated uncertainty between input dimensions** for observed training inputs.

For observation `i`:

```text
x_i,true ~ Normal(mu_i, Sigma_i)
y_i = f(x_i,true) + epsilon_i
```

where `Sigma_i` is a full positive-semidefinite covariance matrix.

This is a surrogate-level uncertainty model. Candidate-time perturbation and
environmental scenarios remain separate compositional layers.

## Why this is a distinct extension

Diagonal `train_X_std` assumes independent location errors across dimensions.
That is often insufficient when process settings, calibration errors, or derived
features share a common source of uncertainty.

For an RBF kernel with metric matrix `Lambda`, the covariance between two
Gaussian input distributions can be evaluated analytically:

```text
S_ij = Sigma_i + Sigma_j

E[k(x_i, x_j)] =
|Lambda|^(1/2) |Lambda + S_ij|^(-1/2)
* exp(
    -1/2 (mu_i - mu_j)^T
    (Lambda + S_ij)^(-1)
    (mu_i - mu_j)
  )
```

The implementation should use stable linear algebra (`cholesky_ex`,
triangular solves, and log-determinants) rather than explicit matrix inverses or
determinants.

## API

Use one uncertainty representation per model construction.

Diagonal uncertainty remains:

```python
UncertainInputSingleTaskGP(
    train_X,
    train_Y,
    train_X_std=train_X_std,
)
```

Full covariance should use:

```python
UncertainInputSingleTaskGP(
    train_X,
    train_Y,
    train_X_covar=train_X_covar,
)
```

with

```text
train_X_covar.shape == train_X.shape[:-1] + (d, d)
```

Exactly one of `train_X_std` and `train_X_covar` must be supplied. Do not
introduce a second compatibility class or an alias.

## Validation

The full covariance input must:

- be finite;
- be symmetric within a documented numerical tolerance;
- be positive semidefinite;
- match the dtype/device and batch/sample shape of `train_X`.

Small numerical jitter may be used only inside stable factorization. It must not
silently redefine invalid covariance matrices as valid.

## Required equivalence tests

1. A diagonal `train_X_covar = diag(train_X_std ** 2)` must reproduce the
   diagonal implementation within numerical tolerance.
2. Zero covariance must recover the deterministic RBF covariance.
3. Nonzero off-diagonal covariance must change the expected kernel relative to
   the diagonal approximation.
4. Batched observations must preserve covariance shape semantics.

## Lifecycle and BoTorch gates

- raw full covariance metadata survives dtype/device migration;
- state-dict roundtrip preserves predictions;
- `make_mll()` remains the exact-GP training contract;
- posterior supports batch/q dimensions;
- qMC acquisition smoke tests remain green;
- candidate-time scenario perturbation remains independently composable.

## Scope boundaries

This model does not add:

- mixed categorical uncertain-input models;
- Reduced/PCA/PLS uncertain-input wrappers;
- multitask uncertain-input wrappers;
- learned input-noise covariance;
- cross-observation covariance between different training rows.

The supplied `Sigma_i` describes uncertainty among feature dimensions for the
same observation.

## Implementation sequence

1. Refactor the Phase 8 kernel representation so diagonal and full covariance
   use one mathematically consistent expected-RBF path where practical.
2. Add `train_X_covar` validation and raw-data retention.
3. Implement stable expected-kernel evaluation.
4. Add diagonal-equivalence, zero-covariance, correlation-sensitivity, shape,
   serialization, dtype/device, posterior, and acquisition tests.
5. Run Ruff lint/format and the Python 3.11/3.12/3.13, package, notebook, and
   optional-dependency CI gates.
