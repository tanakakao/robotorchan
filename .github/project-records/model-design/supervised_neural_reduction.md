# Supervised neural input reduction

Phase 6 adds `SupervisedAutoEncoderInputReducer`, a supervised nonlinear input reducer for high-dimensional continuous inputs.

## Objective

The reducer learns a latent representation `Z = encoder(X)` using two losses:

```text
L = L_reconstruction + supervised_weight * L_supervised
```

`L_reconstruction` is the MSE between standardized inputs and their decoder reconstruction. `L_supervised` is the MSE between standardized outcomes and predictions from a linear auxiliary head attached to the latent representation.

Unlike `AutoEncoderInputReducer`, the representation therefore uses `Y` during pretraining. Unlike joint DKL, the encoder is still frozen after pretraining and is not optimized by the GP marginal likelihood.

## Usage

```python
from robotorchan.reduction import (
    SupervisedAutoEncoderInputReducer,
)

reducer = SupervisedAutoEncoderInputReducer(
    latent_dim=4,
    hidden_dims=(64, 32),
    supervised_weight=1.0,
    epochs=200,
)
reducer.fit(train_X, train_Y)
train_Z = reducer.transform(train_X)
```

The reducer accepts scalar or multi-output regression targets. Outcomes are standardized by default so the auxiliary loss is less sensitive to output scale.

## Lifecycle

The encoder, decoder, and auxiliary prediction head are frozen after fitting. `transform(X)` remains differentiable with respect to `X`, which is required when the reducer is later attached to a `ReducedGP` and acquisition functions are optimized in the original input space.

The auxiliary head is a representation-learning aid rather than the final predictive model. `predict_auxiliary(X)` is exposed mainly for diagnostics. Phase 7 will add the public `SupervisedAutoEncoderGP` wrapper that uses the frozen supervised representation as the GP input.

## Relation to other reducers

- PCA-GP: unsupervised linear representation.
- PLS-GP: supervised linear representation.
- AE-GP: unsupervised nonlinear representation.
- Supervised AE-GP: supervised nonlinear representation with frozen pretraining.
- DKL / joint encoder-GP: supervised nonlinear representation learned jointly through the GP objective; planned separately because its training lifecycle differs.
