import inspect

import torch
from botorch.models import KroneckerMultiTaskGP as BoTorchKroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models.multitask import KroneckerMultiTaskGP


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
