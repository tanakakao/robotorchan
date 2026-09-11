import inspect

import pytest
import torch
from botorch.models import MixedSingleTaskGP as BoTorchMixedSingleTaskGP
from gpytorch.kernels import AdditiveKernel, Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedSingleTaskGP
from robotorchan.models.base import (
    _get_cont_dims,
    _make_mixed_covar_module,
    _normalize_cat_dims,
)


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


def test_normalize_cat_dims_supports_negative_indices_and_stable_order() -> None:
    assert _normalize_cat_dims(cat_dims=[-1, 1], input_dim=4) == [1, 3]
    assert _get_cont_dims(input_dim=4, cat_dims=[-1, 1]) == [0, 2]


@pytest.mark.parametrize(
    ("cat_dims", "input_dim", "message"),
    [
        ([], 3, "at least one"),
        ([3], 3, "out of range"),
        ([-4], 3, "out of range"),
        ([1, -2], 3, "duplicate"),
    ],
)
def test_normalize_cat_dims_rejects_invalid_indices(
    cat_dims: list[int], input_dim: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _normalize_cat_dims(cat_dims=cat_dims, input_dim=input_dim)


def test_make_mixed_covar_module_builds_mixed_kernel_and_passes_active_dims() -> None:
    calls: list[tuple[torch.Size, int, list[int]]] = []

    def factory(batch_shape: torch.Size, ard_num_dims: int, active_dims: list[int]) -> Kernel:
        calls.append((batch_shape, ard_num_dims, active_dims))
        from gpytorch.kernels import RBFKernel

        return RBFKernel(
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
        )

    kernel = _make_mixed_covar_module(
        input_dim=4,
        cat_dims=[-1, 1],
        batch_shape=torch.Size([2]),
        cont_kernel_factory=factory,
    )

    assert isinstance(kernel, AdditiveKernel)
    assert calls == [
        (torch.Size([2]), 2, [0, 2]),
        (torch.Size([2]), 2, [0, 2]),
    ]


def test_make_mixed_covar_module_supports_categorical_only_inputs() -> None:
    kernel = _make_mixed_covar_module(input_dim=2, cat_dims=[0, 1])

    assert isinstance(kernel, ScaleKernel)

    X = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.double)
    covariance = kernel(X).to_dense()

    assert covariance.shape == torch.Size([2, 2])
    assert covariance[0, 0] > covariance[0, 1]
