import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
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


@pytest.mark.parametrize("model_class", [StudentTMultiTaskGP, ContaminatedMultiTaskGP])
def test_heavy_tail_multitask_supports_sampling_and_mc_acquisition(model_class) -> None:
    train_x, train_y = _data()
    model = model_class(train_x, train_y, task_feature=-1, num_inducing=4)
    model.eval()

    candidates = train_x[:2]
    posterior = model.posterior(candidates)
    samples = posterior.rsample(torch.Size([4]))
    objective = GenericMCObjective(lambda values, X=None: values.squeeze(-1))
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    value = acquisition(candidates.unsqueeze(0))

    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert samples.shape == torch.Size([4, 2, 1])
    assert torch.isfinite(samples).all()
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


@pytest.mark.parametrize(
    "model_class",
    [MixedStudentTMultiTaskGP, MixedContaminatedMultiTaskGP],
)
def test_mixed_heavy_tail_supports_sampling_and_mc_acquisition(model_class) -> None:
    train_x, train_y = _data(mixed=True)
    model = model_class(
        train_x,
        train_y,
        task_feature=-1,
        cat_dims=[1],
        num_inducing=4,
    )
    model.eval()

    candidates = train_x[:2]
    posterior = model.posterior(candidates)
    samples = posterior.rsample(torch.Size([4]))
    objective = GenericMCObjective(lambda values, X=None: values.squeeze(-1))
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    value = acquisition(candidates.unsqueeze(0))

    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(samples).all()
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
