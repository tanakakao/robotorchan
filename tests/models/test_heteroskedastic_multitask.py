import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import AdditiveKernel, ProductKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    HeteroskedasticMultiTaskGP,
    MixedHeteroskedasticMultiTaskGP,
)
from robotorchan.models.robust.robust import (
    HeteroskedasticKroneckerMultiTaskGP,
    MixedHeteroskedasticKroneckerMultiTaskGP,
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
    with pytest.raises(ValueError, match="cat_dims must not overlap structural dimensions"):
        MixedHeteroskedasticMultiTaskGP(
            train_x,
            train_y,
            task_feature=-1,
            cat_dims=[-1],
        )


def test_heteroskedastic_multitask_supports_sampling_and_mc_acquisition() -> None:
    train_x, train_y = _long_format_data()
    model = HeteroskedasticMultiTaskGP(train_x, train_y, task_feature=-1)
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


def test_mixed_heteroskedastic_multitask_supports_mixed_optimizer() -> None:
    x = torch.tensor([[0.1, 0.0], [0.4, 1.0], [0.7, 0.0]], dtype=torch.double)
    task0 = torch.zeros(3, 1, dtype=torch.double)
    task1 = torch.ones(3, 1, dtype=torch.double)
    train_x = torch.cat((torch.cat((x, task0), dim=-1), torch.cat((x, task1), dim=-1)))
    train_y = torch.cat((torch.sin(x[:, :1]), torch.cos(x[:, :1])))
    model = MixedHeteroskedasticMultiTaskGP(train_x, train_y, task_feature=-1, cat_dims=[1])
    model.eval()

    objective = GenericMCObjective(lambda values, X=None: values.squeeze(-1))
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    candidate, value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=8,
        fixed_features_list=[
            {1: category, 2: task} for category in (0.0, 1.0) for task in (0.0, 1.0)
        ],
    )

    assert candidate.shape == torch.Size([1, 3])
    assert candidate[0, 1].item() in {0.0, 1.0}
    assert candidate[0, 2].item() in {0.0, 1.0}
    assert torch.isfinite(value).all()


def test_heteroskedastic_kronecker_preserves_block_design_contract() -> None:
    train_x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    train_y = torch.cat((torch.sin(train_x), torch.cos(train_x)), dim=-1)
    model = HeteroskedasticKroneckerMultiTaskGP(train_x, train_y)

    torch.testing.assert_close(model.raw_train_X, train_x)
    torch.testing.assert_close(model.raw_train_Y, train_y)
    assert model.noise_model is None
    assert model.make_mll() is not None
    with pytest.raises(RuntimeError, match="fit_heteroskedastic"):
        model.predicted_noise(train_x)


def test_heteroskedastic_kronecker_supports_sampling_and_mc_acquisition() -> None:
    train_x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    train_y = torch.cat((torch.sin(train_x), torch.cos(train_x)), dim=-1)
    model = HeteroskedasticKroneckerMultiTaskGP(train_x, train_y)
    model.eval()

    test_x = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = model.posterior(test_x)
    samples = posterior.rsample(torch.Size([4]))
    weights = torch.tensor([0.6, 0.4], dtype=torch.double)
    objective = GenericMCObjective(lambda values, X=None: values @ weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
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


def test_mixed_heteroskedastic_kronecker_preserves_mixed_block_design() -> None:
    train_x = torch.tensor(
        [[0.1, 0.0], [0.4, 1.0], [0.7, 0.0], [0.9, 1.0]],
        dtype=torch.double,
    )
    train_y = torch.cat((torch.sin(train_x[:, :1]), torch.cos(train_x[:, :1])), dim=-1)
    model = MixedHeteroskedasticKroneckerMultiTaskGP(
        train_x,
        train_y,
        cat_dims=[-1],
    )
    model.eval()

    posterior = model.posterior(train_x[:2])
    samples = posterior.rsample(torch.Size([4]))

    assert model.cat_dims == (1,)
    torch.testing.assert_close(model.raw_train_X, train_x)
    torch.testing.assert_close(model.raw_train_Y, train_y)
    assert posterior.mean.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([4, 2, 2])
    assert torch.isfinite(samples).all()


def test_mixed_heteroskedastic_kronecker_supports_mc_acquisition() -> None:
    train_x = torch.tensor(
        [[0.1, 0.0], [0.4, 1.0], [0.7, 0.0], [0.9, 1.0]],
        dtype=torch.double,
    )
    train_y = torch.cat((torch.sin(train_x[:, :1]), torch.cos(train_x[:, :1])), dim=-1)
    model = MixedHeteroskedasticKroneckerMultiTaskGP(train_x, train_y, cat_dims=[1])
    model.eval()

    weights = torch.tensor([0.6, 0.4], dtype=torch.double)
    objective = GenericMCObjective(lambda values, X=None: values @ weights)
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=(train_y @ weights).max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        objective=objective,
    )
    value = acquisition(train_x[:2].unsqueeze(0))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()
