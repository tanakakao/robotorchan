import pytest
import torch
from gpytorch.kernels import AdditiveKernel, ProductKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    HeteroskedasticMultiTaskGP,
    MixedHeteroskedasticMultiTaskGP,
)


def _long_format_data() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([[0.1], [0.4], [0.7]], dtype=torch.double)
    task0 = torch.zeros(3, 1, dtype=torch.double)
    task1 = torch.ones(3, 1, dtype=torch.double)
    train_x = torch.cat((torch.cat((x, task0), dim=-1), torch.cat((x, task1), dim=-1)))
    train_y = torch.cat((torch.sin(x), torch.cos(x)))
    return train_x, train_y


def test_heteroskedastic_multitask_public_contract() -> None:
    train_x, train_y = _long_format_data()
    model = HeteroskedasticMultiTaskGP(train_x, train_y, task_feature=-1)

    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    torch.testing.assert_close(model.raw_train_X, train_x)
    torch.testing.assert_close(model.raw_train_Y, train_y)
    assert model.raw_train_Yvar is not None
    assert model.noise_model is None

    with pytest.raises(RuntimeError, match="fit_heteroskedastic"):
        model.predicted_noise(train_x)


def test_heteroskedastic_multitask_validates_noise_floor() -> None:
    train_x, train_y = _long_format_data()
    with pytest.raises(ValueError, match="noise_floor"):
        HeteroskedasticMultiTaskGP(train_x, train_y, task_feature=-1, noise_floor=0.0)


def test_mixed_heteroskedastic_multitask_preserves_task_feature() -> None:
    x = torch.tensor([[0.1, 0.0], [0.4, 1.0], [0.7, 0.0]], dtype=torch.double)
    task0 = torch.zeros(3, 1, dtype=torch.double)
    task1 = torch.ones(3, 1, dtype=torch.double)
    train_x = torch.cat((torch.cat((x, task0), dim=-1), torch.cat((x, task1), dim=-1)))
    train_y = torch.cat((torch.sin(x[:, :1]), torch.cos(x[:, :1])))

    model = MixedHeteroskedasticMultiTaskGP(
        train_x,
        train_y,
        task_feature=-1,
        cat_dims=[1],
    )

    assert model.cat_dims == (1,)
    assert isinstance(model.covar_module, ProductKernel)
    assert isinstance(model.covar_module.kernels[0], AdditiveKernel)
    assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)


def test_mixed_heteroskedastic_multitask_rejects_task_as_category() -> None:
    train_x, train_y = _long_format_data()
    with pytest.raises(ValueError, match="task_feature"):
        MixedHeteroskedasticMultiTaskGP(
            train_x,
            train_y,
            task_feature=-1,
            cat_dims=[-1],
        )
