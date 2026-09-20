import pytest
import torch
from gpytorch.kernels import ProductKernel

from robotorchan.models import (
    MixedNonstationaryMultiTaskGP,
    NonstationaryMultiTaskGP,
)


def _data(mixed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([[0.1], [0.4], [0.7]], dtype=torch.double)
    if mixed:
        category = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
        x = torch.cat((x, category), dim=-1)
    task0 = torch.zeros(3, 1, dtype=torch.double)
    task1 = torch.ones(3, 1, dtype=torch.double)
    train_x = torch.cat((torch.cat((x, task0), dim=-1), torch.cat((x, task1), dim=-1)))
    train_y = torch.cat((torch.sin(x[:, :1]), torch.cos(x[:, :1])))
    return train_x, train_y


def test_nonstationary_multitask_contract() -> None:
    train_x, train_y = _data()
    model = NonstationaryMultiTaskGP(train_x, train_y, task_feature=-1)

    assert model.task_feature == 1
    assert model.continuous_dims == (0,)
    assert isinstance(model.covar_module, ProductKernel)
    lengthscale = model.local_lengthscale(train_x)
    assert lengthscale.shape == (6, 1)
    assert torch.all(lengthscale > 0)
    assert model.make_mll() is not None


def test_mixed_nonstationary_multitask_contract() -> None:
    train_x, train_y = _data(mixed=True)
    model = MixedNonstationaryMultiTaskGP(
        train_x,
        train_y,
        task_feature=-1,
        cat_dims=[1],
    )

    assert model.task_feature == 2
    assert model.cat_dims == (1,)
    assert model.continuous_dims == (0,)
    additive, interaction = model.local_lengthscale(train_x)
    assert additive.shape == interaction.shape == (6, 1)
    assert torch.all(additive > 0)
    assert torch.all(interaction > 0)


def test_mixed_nonstationary_multitask_rejects_task_as_category() -> None:
    train_x, train_y = _data()
    with pytest.raises(ValueError, match="cat_dims must not overlap structural dimensions"):
        MixedNonstationaryMultiTaskGP(
            train_x,
            train_y,
            task_feature=-1,
            cat_dims=[-1],
        )
