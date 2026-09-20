import pytest
import torch
from botorch.models.likelihoods.sparse_outlier_noise import SparseOutlierGaussianLikelihood
from gpytorch.kernels import AdditiveKernel, ProductKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedRobustRelevancePursuitMultiTaskGP,
    RobustRelevancePursuitMultiTaskGP,
)


def _long_format_data() -> tuple[torch.Tensor, torch.Tensor]:
    data_x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    task0 = torch.zeros(4, 1, dtype=torch.double)
    task1 = torch.ones(4, 1, dtype=torch.double)
    train_x = torch.cat(
        (torch.cat((data_x, task0), dim=-1), torch.cat((data_x, task1), dim=-1))
    )
    base = torch.sin(data_x[:, 0])
    train_y = torch.cat((base, 0.8 * base + 0.1)).unsqueeze(-1)
    return train_x, train_y


def test_robust_multitask_preserves_public_contract() -> None:
    train_x, train_y = _long_format_data()
    model = RobustRelevancePursuitMultiTaskGP(train_x, train_y, task_feature=-1)

    assert isinstance(model.likelihood, SparseOutlierGaussianLikelihood)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    torch.testing.assert_close(model.raw_train_X, train_x)
    torch.testing.assert_close(model.raw_train_Y, train_y)

    model.eval()
    model.likelihood.eval()
    posterior = model.posterior(train_x[:2])
    assert posterior.mean.shape[-1] == 1
    assert torch.isfinite(posterior.mean).all()


def test_robust_multitask_standard_model_keeps_structure() -> None:
    train_x, train_y = _long_format_data()
    model = RobustRelevancePursuitMultiTaskGP(train_x, train_y, task_feature=-1)
    standard = model.to_standard_model()

    assert not isinstance(standard, RobustRelevancePursuitMultiTaskGP)
    assert standard._task_feature == 1
    assert standard.likelihood is model.likelihood


def test_mixed_robust_multitask_preserves_task_feature() -> None:
    data_x = torch.tensor(
        [[0.1, 0.0], [0.4, 1.0], [0.7, 0.0], [0.9, 1.0]],
        dtype=torch.double,
    )
    task0 = torch.zeros(4, 1, dtype=torch.double)
    task1 = torch.ones(4, 1, dtype=torch.double)
    train_x = torch.cat(
        (torch.cat((data_x, task0), dim=-1), torch.cat((data_x, task1), dim=-1))
    )
    base = torch.sin(data_x[:, 0])
    train_y = torch.cat((base, 0.8 * base + 0.1)).unsqueeze(-1)

    model = MixedRobustRelevancePursuitMultiTaskGP(
        train_x,
        train_y,
        task_feature=-1,
        cat_dims=[1],
    )

    assert model.cat_dims == (1,)
    assert isinstance(model.covar_module, ProductKernel)
    assert isinstance(model.covar_module.kernels[0], AdditiveKernel)
    assert isinstance(model.likelihood, SparseOutlierGaussianLikelihood)
    torch.testing.assert_close(model.raw_train_X, train_x)


def test_mixed_robust_multitask_rejects_task_as_category() -> None:
    train_x, train_y = _long_format_data()
    with pytest.raises(ValueError):
        MixedRobustRelevancePursuitMultiTaskGP(
            train_x,
            train_y,
            task_feature=-1,
            cat_dims=[-1],
        )
