import inspect

import torch
from botorch.models import KroneckerMultiTaskGP as BoTorchKroneckerMultiTaskGP
from gpytorch.kernels import AdditiveKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models.multitask import KroneckerMultiTaskGP, MixedKroneckerMultiTaskGP


def _make_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_X = torch.rand(10, 2, dtype=torch.double)
    train_Y = torch.cat(
        [
            torch.sin(train_X[:, :1] * 2.0) + train_X[:, 1:2],
            torch.cos(train_X[:, :1] * 1.5) - 0.5 * train_X[:, 1:2],
            train_X[:, :1].square() + 0.2 * train_X[:, 1:2],
        ],
        dim=-1,
    )
    return train_X, train_Y


def _make_mixed_data() -> tuple[torch.Tensor, torch.Tensor]:
    continuous = torch.rand(12, 2, dtype=torch.double)
    categorical = torch.randint(0, 3, (12, 1)).to(dtype=torch.double)
    train_X = torch.cat([continuous, categorical], dim=-1)
    train_Y = torch.cat(
        [
            torch.sin(continuous[:, :1] * 2.0) + 0.2 * categorical,
            torch.cos(continuous[:, :1] * 1.5) + continuous[:, 1:2] + 0.1 * categorical,
            continuous[:, :1].square() - 0.3 * continuous[:, 1:2] + 0.15 * categorical,
        ],
        dim=-1,
    )
    return train_X, train_Y


def test_kronecker_multi_task_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(KroneckerMultiTaskGP.__init__)
    upstream = inspect.signature(BoTorchKroneckerMultiTaskGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_kronecker_multi_task_gp_uses_common_wrapper_contract() -> None:
    train_X, train_Y = _make_data()
    model = KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y)

    assert isinstance(model, BoTorchKroneckerMultiTaskGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert model.raw_train_Yvar is None
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_kronecker_multi_task_gp_matches_upstream_posterior() -> None:
    train_X, train_Y = _make_data()
    wrapper = KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y)
    upstream = BoTorchKroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y)

    upstream_state = {
        name: value for name, value in wrapper.state_dict().items() if not name.startswith("_raw_")
    }
    upstream.load_state_dict(upstream_state)

    test_X = torch.rand(4, 2, dtype=torch.double)

    wrapper.eval()
    upstream.eval()
    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)


def test_mixed_kronecker_multi_task_gp_uses_common_wrapper_contract() -> None:
    train_X, train_Y = _make_mixed_data()
    model = MixedKroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=[-1])

    assert isinstance(model, BoTorchKroneckerMultiTaskGP)
    assert model.supports_mll is True
    assert model.cat_dims == (2,)
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert model.raw_train_Yvar is None
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    assert isinstance(model.data_covar_module, AdditiveKernel)


def test_mixed_kronecker_multi_task_gp_posterior_supports_mixed_features() -> None:
    train_X, train_Y = _make_mixed_data()
    model = MixedKroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=[2])

    test_continuous = torch.rand(4, 2, dtype=torch.double)
    test_categorical = torch.randint(0, 3, (4, 1)).to(dtype=torch.double)
    test_X = torch.cat([test_continuous, test_categorical], dim=-1)

    model.eval()
    posterior = model.posterior(test_X)

    assert posterior.mean.shape == torch.Size([4, 3])
    assert posterior.variance.shape == torch.Size([4, 3])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mixed_kronecker_multi_task_gp_supports_categorical_only_inputs() -> None:
    train_X = torch.randint(0, 3, (10, 2)).to(dtype=torch.double)
    train_Y = torch.stack(
        [train_X[:, 0] + 0.1 * train_X[:, 1], train_X[:, 1] - 0.2 * train_X[:, 0]],
        dim=-1,
    )

    model = MixedKroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=[0, 1])

    assert model.cat_dims == (0, 1)
    assert isinstance(model.data_covar_module, ScaleKernel)
