# Mixed × SAAS feasibility

## Current implementation

robotorchan now exposes fully Bayesian mixed SAAS wrappers:

- `MixedSaasFullyBayesianSingleTaskGP`
- `MixedSaasFullyBayesianMultiTaskGP`

They use model-owned categorical one-hot encoding before BoTorch's fully Bayesian
SAAS model. In the multi-task wrapper, the task feature remains structural and is
excluded from categorical encoding. Fitting uses BoTorch's
`fit_fully_bayesian_model_nuts`; these models intentionally do not expose MLL fitting.

This is different from constructing a `MixedSingleTaskGP` categorical kernel with a
SAAS-flavoured continuous kernel. robotorchan does not expose that composition as a
SAAS model because it would not preserve BoTorch SAAS inference semantics.

## High-dimensional guidance

Use fully Bayesian mixed SAAS when categorical variables must be represented explicitly
and sparsity over the encoded continuous/one-hot coordinate space is acceptable. Use
mixed reduced models when the intended assumption is instead a low-dimensional latent
representation of the continuous variables.

Do not add a `ReducedSaas...` or `Saas...Kronecker...` class by naming convention alone.
Those combinations require a separately defined prior, covariance, and inference model.
They must not be compatibility aliases or thin renames of existing models.
