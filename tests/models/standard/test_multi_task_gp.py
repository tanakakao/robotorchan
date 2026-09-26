import inspect

import torch
from botorch.models import MultiTaskGP as BoTorchMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models.standard.multitask import MultiTaskGP


def _make_data() -> tuple[torch.Tensor, torch.Tensor]:
    task_0_X = torch.rand(7, 2, dtype=torch.double)
    task_1_X = torch.rand(7, 2, dtype=torch.double)
    task_0 = torch.zeros(7, 1, dtype=torch.double)
    task_1 = torch.ones(7, 1, dtype=torch.double)

    train_X = torch.cat(
        [
            torch.cat([task_0_X, task_0], dim=-1),
            torch.cat([task_1_X, task_1], dim=-1),
        ],
        dim=0,
    )
    train_Y = torch.cat(
        [
            torch.sin(task_0_X[:, :1] * 2.0) + task_0_X[:, 1:2],
            torch.sin(task_1_X[:, :1] * 2.0) + task_1_X[:, 1:2] + 0.4,
        ],
        dim=0,
    )
    return train_X, train_Y


def test_multi_task_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(MultiTaskGP.__init__)
    upstream = inspect.signature(BoTorchMultiTaskGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_multi_task_gp_uses_common_wrapper_contract() -> None:
    train_X, train_Y = _make_data()
    train_Yvar = torch.full_like(train_Y, 1e-4)

    model = MultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        train_Yvar=train_Yvar,
    )

    assert isinstance(model, BoTorchMultiTaskGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert torch.equal(model.raw_train_Yvar, train_Yvar)
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_multi_task_gp_matches_upstream_posterior() -> None:
    train_X, train_Y = _make_data()
    wrapper = MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=-1)
    upstream = BoTorchMultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=-1)

    upstream_state = {
        name: value for name, value in wrapper.state_dict().items() if not name.startswith("_raw_")
    }
    upstream.load_state_dict(upstream_state)

    test_X = torch.rand(5, 2, dtype=torch.double)

    wrapper.eval()
    upstream.eval()
    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)
