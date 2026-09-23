import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from gpytorch.kernels import ProductKernel

from robotorchan.models import (
    MixedNonstationaryMultiTaskGP,
    NonstationaryKroneckerMultiTaskGP,
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


def _assert_posterior_and_acquisition(model, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
    model.eval()
    test_x = train_x[:2].clone()
    posterior = model.posterior(test_x)
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

    objective = GenericMCObjective(lambda samples, X=None: samples.squeeze(-1))
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        objective=objective,
    )
    value = acquisition(test_x.unsqueeze(0))
    assert torch.isfinite(value).all()


def test_nonstationary_multitask_posterior_and_acquisition() -> None:
    train_x, train_y = _data()
    model = NonstationaryMultiTaskGP(train_x, train_y, task_feature=-1)
    _assert_posterior_and_acquisition(model, train_x, train_y)


def test_mixed_nonstationary_multitask_posterior_and_acquisition() -> None:
    train_x, train_y = _data(mixed=True)
    model = MixedNonstationaryMultiTaskGP(
        train_x,
        train_y,
        task_feature=-1,
        cat_dims=[1],
    )
    _assert_posterior_and_acquisition(model, train_x, train_y)


def test_nonstationary_kronecker_supports_sampling_and_mc_acquisition() -> None:
    train_x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    train_y = torch.cat((torch.sin(train_x), torch.cos(train_x)), dim=-1)
    model = NonstationaryKroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    test_x = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(test_x)
    samples = posterior.rsample(torch.Size([4]))
    weights = torch.tensor([0.6, 0.4], dtype=torch.double)
    objective = GenericMCObjective(lambda values, X=None: values @ weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        objective=objective,
    )
    value = acquisition(test_x.unsqueeze(0))

    assert posterior.mean.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([4, 2, 2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(samples).all()
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
