import inspect

import pytest
import torch
from botorch.models import SingleTaskVariationalGP as BoTorchSingleTaskVariationalGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import VariationalELBO

from robotorchan.models import SingleTaskVariationalGP


def test_variational_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(SingleTaskVariationalGP.__init__)
    upstream = inspect.signature(BoTorchSingleTaskVariationalGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_variational_gp_retains_raw_training_data() -> None:
    train_X = torch.rand(12, 3, dtype=torch.double)
    train_Y = 5.0 + 2.0 * train_X[:, :1]
    original_Y = train_Y.clone()

    model = SingleTaskVariationalGP(
        train_X=train_X,
        train_Y=train_Y,
        inducing_points=4,
        outcome_transform=Standardize(m=1),
    )

    assert isinstance(model, BoTorchSingleTaskVariationalGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, original_Y)
    assert model.raw_train_Yvar is None
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()
    assert set(model.raw_data) == {"train_X", "train_Y", "train_Yvar"}


def test_variational_gp_allows_missing_train_y() -> None:
    train_X = torch.rand(10, 2, dtype=torch.double)

    model = SingleTaskVariationalGP(train_X=train_X, inducing_points=3)

    assert torch.equal(model.raw_train_X, train_X)
    assert model.raw_train_Y is None
    assert model.raw_train_Yvar is None


def test_variational_gp_make_mll_uses_raw_training_size_by_default() -> None:
    train_X = torch.rand(15, 2, dtype=torch.double)
    train_Y = train_X.sin().sum(dim=-1, keepdim=True)
    model = SingleTaskVariationalGP(
        train_X=train_X,
        train_Y=train_Y,
        inducing_points=5,
    )

    mll = model.make_mll()

    assert isinstance(mll, VariationalELBO)
    assert mll.model is model.model
    assert mll.likelihood is model.likelihood
    assert mll.num_data == train_X.shape[-2]


def test_variational_gp_make_mll_accepts_total_data_size_override() -> None:
    minibatch_X = torch.rand(8, 2, dtype=torch.double)
    minibatch_Y = minibatch_X[:, :1]
    model = SingleTaskVariationalGP(
        train_X=minibatch_X,
        train_Y=minibatch_Y,
        inducing_points=4,
    )

    mll = model.make_mll(num_data=128)

    assert mll.num_data == 128


def test_variational_gp_make_mll_rejects_non_positive_num_data() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    model = SingleTaskVariationalGP(train_X=train_X, inducing_points=4)

    with pytest.raises(ValueError, match="num_data must be positive"):
        model.make_mll(num_data=0)


def test_variational_gp_raw_buffers_follow_dtype_and_are_non_persistent() -> None:
    train_X = torch.rand(10, 2, dtype=torch.double)
    train_Y = train_X[:, :1]
    model = SingleTaskVariationalGP(
        train_X=train_X,
        train_Y=train_Y,
        inducing_points=4,
    )

    state_dict = model.state_dict()
    model = model.to(dtype=torch.float32)

    assert "_raw_train_X" not in state_dict
    assert "_raw_train_Y" not in state_dict
    assert "_raw_train_Yvar" not in state_dict
    assert model.raw_train_X.dtype == torch.float32
    assert model.raw_train_Y.dtype == torch.float32


def test_variational_gp_matches_upstream_posterior() -> None:
    train_X = torch.rand(14, 2, dtype=torch.double)
    train_Y = torch.sin(train_X[:, :1] * 2.0)
    inducing_points = train_X[:5].clone()
    test_X = torch.rand(6, 2, dtype=torch.double)

    wrapper = SingleTaskVariationalGP(
        train_X=train_X,
        train_Y=train_Y,
        inducing_points=inducing_points,
    )
    upstream = BoTorchSingleTaskVariationalGP(
        train_X=train_X,
        train_Y=train_Y,
        inducing_points=inducing_points.clone(),
    )

    wrapper.eval()
    wrapper.posterior(test_X)
    upstream.model.load_state_dict(wrapper.model.state_dict())
    upstream.likelihood.load_state_dict(wrapper.likelihood.state_dict())
    upstream.eval()

    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)
