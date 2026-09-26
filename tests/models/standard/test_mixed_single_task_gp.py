import inspect

import torch
from botorch.models import MixedSingleTaskGP as BoTorchMixedSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedSingleTaskGP


def _make_data() -> tuple[torch.Tensor, torch.Tensor]:
    continuous = torch.rand(12, 2, dtype=torch.double)
    categorical = torch.randint(0, 3, (12, 1)).to(dtype=torch.double)
    train_X = torch.cat([continuous, categorical], dim=-1)
    train_Y = torch.sin(continuous[:, :1] * 2.0) + 0.2 * categorical
    return train_X, train_Y


def test_mixed_single_task_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(MixedSingleTaskGP.__init__)
    upstream = inspect.signature(BoTorchMixedSingleTaskGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_mixed_single_task_gp_uses_common_wrapper_contract() -> None:
    train_X, train_Y = _make_data()
    train_Yvar = torch.full_like(train_Y, 1e-4)

    model = MixedSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        cat_dims=[-1],
        train_Yvar=train_Yvar,
    )

    assert isinstance(model, BoTorchMixedSingleTaskGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert torch.equal(model.raw_train_Yvar, train_Yvar)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_single_task_gp_matches_upstream_posterior() -> None:
    train_X, train_Y = _make_data()
    wrapper = MixedSingleTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=[-1])
    upstream = BoTorchMixedSingleTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=[-1])

    upstream_state = {
        name: value for name, value in wrapper.state_dict().items() if not name.startswith("_raw_")
    }
    upstream.load_state_dict(upstream_state)

    test_continuous = torch.rand(5, 2, dtype=torch.double)
    test_categorical = torch.randint(0, 3, (5, 1)).to(dtype=torch.double)
    test_X = torch.cat([test_continuous, test_categorical], dim=-1)

    wrapper.eval()
    upstream.eval()
    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)
