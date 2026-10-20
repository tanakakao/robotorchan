# DeepGP models and input contracts

robotorchan provides `SingleTaskDeepGP`, `MixedSingleTaskDeepGP`, and
`MultiTaskDeepGP`. These are variational Deep Gaussian Process models,
not exact GP wrappers. Choose the variant from the input and task structure.

## Single-task continuous inputs

`SingleTaskDeepGP` uses a hierarchy of stochastic GP layers for
single-task regression. It exposes a Monte Carlo posterior for BoTorch
acquisition integration; stochastic DeepGP trajectories must not be
interpreted as an exact Gaussian posterior.

## Mixed continuous and categorical inputs

`MixedSingleTaskDeepGP` accepts the original mixed input space and
requires `cat_dims`. It standardizes continuous features while encoding
categorical features with **learned embeddings** before the DeepGP
layers. Integer category IDs are not treated as continuous distances.

Categorical feature indices may be negative. Category values must be
non-negative, integer-valued, contiguous, and zero-based in training
data. At least one continuous feature is required. The learned
embedding is a model-specific representation, not the exact-GP
`MixedSingleTaskGP` categorical kernel.

## Multi-task long-format inputs

`MultiTaskDeepGP` accepts a long-format task feature and embeds task
identity separately from standardized data features. Pass original
long-format inputs to `posterior()` and `training_loss()`.
This is distinct from block-design `train_Y[n, m]` models.

## Training and acquisition integration

DeepGP training uses a variational objective rather than the exact
GP marginal likelihood fitting path. Posterior sampling preserves
stochastic layers and supports batched candidate evaluation under
the model's tested shape contracts. Verify acquisition compatibility
and the chosen optimization backend before assuming a specific
candidate-search path is supported.

See [Expressive model families](expressive.md) and
[Expressive BoTorch integration](expressive_botorch_integration.md)
for model-family selection and integration boundaries.
