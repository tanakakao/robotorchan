import inspect

import torch
from botorch.models.latent_kronecker_gp import (
    LatentKroneckerGP as BoTorchLatentKroneckerGP,
)
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models.latent_kronecker import LatentKroneckerGP


def _training_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.10, 0.20],
            [0.35, 0.45],
            [0.60, 0.55],
            [0.85, 0.80],
        ],
        dtype=torch.double,
    )
    train_T = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    base = train_X[:, :1] + train_X[:, 1:2]
    train_Y = base + train_T.squeeze(-1)
    return train_X, train_T, train_Y


def test_latent_kronecker_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(LatentKroneckerGP.__init__)
    upstream = inspect.signature(BoTorchLatentKroneckerGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_latent_kronecker_gp_retains_product_space_raw_data() -> None:
    train_X, train_T, train_Y = _training_data()
    train_Y = train_Y.clone()
    train_Y[1, 1] = torch.nan
    original_Y = train_Y.clone()

    model = LatentKroneckerGP(
        train_X=train_X,
        train_T=train_T,
        train_Y=train_Y,
        input_transform=Normalize(d=2),
    )

    assert isinstance(model, BoTorchLatentKroneckerGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_T, train_T)
    torch.testing.assert_close(model.raw_train_Y, original_Y, equal_nan=True)
    assert model.raw_train_Yvar is None
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_T.data_ptr() != train_T.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()
    assert set(model.raw_data) == {"train_X", "train_T", "train_Y", "train_Yvar"}
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_latent_kronecker_gp_raw_buffers_follow_dtype_and_serialize() -> None:
    train_X, train_T, train_Y = _training_data()
    model = LatentKroneckerGP(train_X=train_X, train_T=train_T, train_Y=train_Y)

    state_dict = model.state_dict()
    model = model.to(dtype=torch.float32)

    assert "_raw_train_X" in state_dict
    assert "_raw_train_T" in state_dict
    assert "_raw_train_Y" in state_dict
    assert "_raw_train_Yvar" not in state_dict
    assert model.raw_train_X.dtype == torch.float32
    assert model.raw_train_T.dtype == torch.float32
    assert model.raw_train_Y.dtype == torch.float32


def test_latent_kronecker_gp_matches_upstream_posterior() -> None:
    train_X, train_T, train_Y = _training_data()
    wrapper = LatentKroneckerGP(train_X=train_X, train_T=train_T, train_Y=train_Y)
    upstream = BoTorchLatentKroneckerGP(
        train_X=train_X,
        train_T=train_T,
        train_Y=train_Y,
    )

    upstream_state = {
        name: value for name, value in wrapper.state_dict().items() if not name.startswith("_raw_")
    }
    upstream.load_state_dict(upstream_state)

    test_X = torch.tensor([[0.20, 0.30], [0.70, 0.65]], dtype=torch.double)
    test_T = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    wrapper.eval()
    upstream.eval()

    with wrapper.use_iterative_methods():
        wrapper_posterior = wrapper.posterior(test_X, test_T)
    with upstream.use_iterative_methods():
        upstream_posterior = upstream.posterior(test_X, test_T)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)
