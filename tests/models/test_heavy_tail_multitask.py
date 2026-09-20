import pytest
import torch
from gpytorch.kernels import ProductKernel

from robotorchan.models import (
    ContaminatedMultiTaskGP,
    MixedContaminatedMultiTaskGP,
    MixedStudentTMultiTaskGP,
    StudentTMultiTaskGP,
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


@pytest.mark.parametrize("model_class", [StudentTMultiTaskGP, ContaminatedMultiTaskGP])
def test_heavy_tail_multitask_contract(model_class) -> None:
    train_x, train_y = _data()
    model = model_class(train_x, train_y, task_feature=-1, num_inducing=4)

    assert model.task_feature == 1
    assert isinstance(model.response_model.model.covar_module, ProductKernel)
    assert model.supports_mll is False
    torch.testing.assert_close(model.raw_train_X, train_x)
    torch.testing.assert_close(model.raw_train_Y, train_y)
    assert torch.isfinite(model.training_loss())


@pytest.mark.parametrize(
    "model_class",
    [MixedStudentTMultiTaskGP, MixedContaminatedMultiTaskGP],
)
def test_mixed_heavy_tail_multitask_contract(model_class) -> None:
    train_x, train_y = _data(mixed=True)
    model = model_class(
        train_x,
        train_y,
        task_feature=-1,
        cat_dims=[1],
        num_inducing=4,
    )

    assert model.task_feature == 2
    assert model.cat_dims == (1,)
    assert isinstance(model.response_model.model.covar_module, ProductKernel)
    assert torch.isfinite(model.training_loss())


@pytest.mark.parametrize(
    "model_class",
    [MixedStudentTMultiTaskGP, MixedContaminatedMultiTaskGP],
)
def test_mixed_heavy_tail_rejects_task_as_category(model_class) -> None:
    train_x, train_y = _data()
    with pytest.raises(ValueError, match="cat_dims must not overlap structural dimensions"):
        model_class(train_x, train_y, task_feature=-1, cat_dims=[-1])
