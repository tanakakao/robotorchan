import pytest
import torch
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedMultiTaskGP


def _make_data(*, task_feature: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    continuous_0 = torch.rand(8, 1, dtype=torch.double)
    continuous_1 = torch.rand(8, 1, dtype=torch.double)
    categorical_0 = torch.randint(0, 3, (8, 1)).to(dtype=torch.double)
    categorical_1 = torch.randint(0, 3, (8, 1)).to(dtype=torch.double)
    task_0 = torch.zeros(8, 1, dtype=torch.double)
    task_1 = torch.ones(8, 1, dtype=torch.double)

    if task_feature in {-1, 2}:
        X0 = torch.cat([continuous_0, categorical_0, task_0], dim=-1)
        X1 = torch.cat([continuous_1, categorical_1, task_1], dim=-1)
    elif task_feature == 1:
        X0 = torch.cat([continuous_0, task_0, categorical_0], dim=-1)
        X1 = torch.cat([continuous_1, task_1, categorical_1], dim=-1)
    else:  # pragma: no cover - helper guard
        raise ValueError("Unsupported task_feature for test data.")

    Y0 = torch.sin(continuous_0 * 2.0) + 0.2 * categorical_0
    Y1 = torch.sin(continuous_1 * 2.0) + 0.2 * categorical_1 + 0.4
    return torch.cat([X0, X1], dim=0), torch.cat([Y0, Y1], dim=0)


def test_mixed_multi_task_gp_uses_common_wrapper_contract() -> None:
    train_X, train_Y = _make_data()
    train_Yvar = torch.full_like(train_Y, 1e-4)

    model = MixedMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        cat_dims=[1],
        train_Yvar=train_Yvar,
    )

    assert model.supports_mll is True
    assert model.cat_dims == (1,)
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert torch.equal(model.raw_train_Yvar, train_Yvar)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_mixed_multi_task_gp_excludes_task_feature_from_data_kernel() -> None:
    train_X, train_Y = _make_data(task_feature=1)
    calls: list[tuple[torch.Size, int, list[int]]] = []

    def factory(batch_shape: torch.Size, ard_num_dims: int, active_dims: list[int]) -> Kernel:
        calls.append((batch_shape, ard_num_dims, active_dims))
        return RBFKernel(
            batch_shape=batch_shape,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
        )

    model = MixedMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=1,
        cat_dims=[-1],
        cont_kernel_factory=factory,
    )

    assert model.cat_dims == (2,)
    assert calls == [
        (torch.Size(), 1, [0]),
        (torch.Size(), 1, [0]),
    ]

    data_kernel = model.covar_module.kernels[0]
    x_task_0 = torch.tensor([[0.25, 0.0, 1.0]], dtype=torch.double)
    x_task_1 = torch.tensor([[0.25, 1.0, 1.0]], dtype=torch.double)
    torch.testing.assert_close(
        data_kernel(x_task_0).to_dense(),
        data_kernel(x_task_1).to_dense(),
    )


def test_mixed_multi_task_gp_posterior_supports_mixed_features() -> None:
    train_X, train_Y = _make_data()
    model = MixedMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        cat_dims=[1],
    )

    model.eval()
    test_X = torch.cat(
        [
            torch.rand(4, 1, dtype=torch.double),
            torch.randint(0, 3, (4, 1)).to(dtype=torch.double),
        ],
        dim=-1,
    )
    posterior = model.posterior(test_X)

    assert posterior.mean.shape == torch.Size([4, 2])
    assert posterior.variance.shape == torch.Size([4, 2])


def test_mixed_multi_task_gp_rejects_task_feature_in_cat_dims() -> None:
    train_X, train_Y = _make_data()

    with pytest.raises(ValueError, match="task_feature"):
        MixedMultiTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=-1,
            cat_dims=[-1],
        )


def test_mixed_multi_task_gp_supports_categorical_only_data_features() -> None:
    categories = torch.tensor([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]], dtype=torch.double)
    task_0 = torch.zeros(6, 1, dtype=torch.double)
    task_1 = torch.ones(6, 1, dtype=torch.double)
    train_X = torch.cat(
        [
            torch.cat([categories, task_0], dim=-1),
            torch.cat([categories, task_1], dim=-1),
        ],
        dim=0,
    )
    train_Y = torch.cat([0.3 * categories, 0.3 * categories + 0.5], dim=0)

    model = MixedMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=-1,
        cat_dims=[0],
    )

    model.eval()
    posterior = model.posterior(torch.tensor([[0.0], [2.0]], dtype=torch.double))
    assert posterior.mean.shape == torch.Size([2, 2])
