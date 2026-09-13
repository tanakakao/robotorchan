from __future__ import annotations

import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import PCAGP, PLSGP, RandomProjectionGP
from robotorchan.models.reduction import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(101)
    train_X = torch.randn(24, 8, dtype=torch.double)
    train_Y = (
        1.2 * train_X[:, :1]
        - 0.8 * train_X[:, 1:2]
        + 0.3 * train_X[:, 2:3].square()
    )
    return train_X, train_Y


def test_pca_reducer_remains_frozen_after_posterior_and_conditioning() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=3)
    assert isinstance(model.input_reducer, PCAInputReducer)
    assert model.input_reducer.components is not None
    assert model.input_reducer.mean is not None

    components = model.input_reducer.components.detach().clone()
    mean = model.input_reducer.mean.detach().clone()

    model.eval()
    model.likelihood.eval()
    model.posterior(torch.randn(4, 8, dtype=torch.double))

    new_X = torch.randn(3, 8, dtype=torch.double)
    new_Y = torch.randn(3, 1, dtype=torch.double)
    conditioned = model.condition_on_observations(X=new_X, Y=new_Y)

    assert isinstance(conditioned.input_reducer, PCAInputReducer)
    assert conditioned.input_reducer.components is not None
    assert conditioned.input_reducer.mean is not None
    torch.testing.assert_close(model.input_reducer.components, components)
    torch.testing.assert_close(model.input_reducer.mean, mean)
    torch.testing.assert_close(conditioned.input_reducer.components, components)
    torch.testing.assert_close(conditioned.input_reducer.mean, mean)


def test_pca_reducer_remains_frozen_in_fantasy_model() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=3)
    assert isinstance(model.input_reducer, PCAInputReducer)
    assert model.input_reducer.components is not None
    assert model.input_reducer.mean is not None

    components = model.input_reducer.components.detach().clone()
    mean = model.input_reducer.mean.detach().clone()

    model.eval()
    model.likelihood.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([4]), seed=11)
    fantasy_model = model.fantasize(
        X=torch.randn(2, 8, dtype=torch.double),
        sampler=sampler,
    )

    assert isinstance(fantasy_model.input_reducer, PCAInputReducer)
    assert fantasy_model.input_reducer.components is not None
    assert fantasy_model.input_reducer.mean is not None
    torch.testing.assert_close(fantasy_model.input_reducer.components, components)
    torch.testing.assert_close(fantasy_model.input_reducer.mean, mean)

    posterior = fantasy_model.posterior(torch.randn(3, 8, dtype=torch.double))
    assert posterior.mean.shape[-2:] == torch.Size([3, 1])


def test_pls_reducer_remains_frozen_after_prediction() -> None:
    train_X, train_Y = _training_data()
    model = PLSGP(train_X=train_X, train_Y=train_Y, n_components=2)
    assert isinstance(model.input_reducer, PLSInputReducer)
    assert model.input_reducer.rotation is not None

    rotation = model.input_reducer.rotation.detach().clone()
    model.posterior(torch.randn(5, 8, dtype=torch.double))

    torch.testing.assert_close(model.input_reducer.rotation, rotation)


def test_random_projection_remains_frozen_after_prediction() -> None:
    train_X, train_Y = _training_data()
    model = RandomProjectionGP(
        train_X=train_X,
        train_Y=train_Y,
        n_components=3,
        random_state=17,
    )
    assert isinstance(model.input_reducer, RandomProjectionInputReducer)
    assert model.input_reducer.projection is not None

    projection = model.input_reducer.projection.detach().clone()
    model.posterior(torch.randn(5, 8, dtype=torch.double))

    torch.testing.assert_close(model.input_reducer.projection, projection)


def test_reducer_buffers_follow_model_dtype_conversion() -> None:
    train_X, train_Y = _training_data()
    model = PCAGP(train_X=train_X, train_Y=train_Y, n_components=3).float()
    assert isinstance(model.input_reducer, PCAInputReducer)
    assert model.input_reducer.components is not None
    assert model.input_reducer.mean is not None

    assert model.input_reducer.components.dtype == torch.float32
    assert model.input_reducer.mean.dtype == torch.float32

    model.eval()
    model.likelihood.eval()
    posterior = model.posterior(torch.randn(4, 8, dtype=torch.float32))
    assert posterior.mean.dtype == torch.float32
