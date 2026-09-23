import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.likelihoods.sparse_outlier_noise import SparseOutlierGaussianLikelihood
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import AdditiveKernel, ProductKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedRobustRelevancePursuitMultiTaskGP,
    MultiTaskGP,
    RobustRelevancePursuitMultiTaskGP,
)


def _long_format_data() -> tuple[torch.Tensor, torch.Tensor]:
    data_x = torch.tensor([[0.1], [0.4], [0.7], [0.9]], dtype=torch.double)
    task0 = torch.zeros(4, 1, dtype=torch.double)
    task1 = torch.ones(4, 1, dtype=torch.double)
    train_x = torch.cat((torch.cat((data_x, task0), dim=-1), torch.cat((data_x, task1), dim=-1)))
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

    standard = model.to_standard_model()
    assert isinstance(standard, MultiTaskGP)
    assert standard.likelihood is model.likelihood
    assert type(standard.covar_module) is type(model.covar_module)
    assert type(standard.mean_module) is type(model.mean_module)


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
    train_x = torch.cat((torch.cat((data_x, task0), dim=-1), torch.cat((data_x, task1), dim=-1)))
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



def test_robust_relevance_pursuit_multitask_supports_mc_acquisition() -> None:
    train_x, train_y = _long_format_data()
    model = RobustRelevancePursuitMultiTaskGP(train_x, train_y, task_feature=-1)
    model.eval()
    acquisition_model = model.to_standard_model()

    candidates = train_x[:2]
    posterior = acquisition_model.posterior(candidates)
    samples = posterior.rsample(torch.Size([4]))
    objective = GenericMCObjective(lambda values, X=None: values.squeeze(-1))
    acquisition = qLogExpectedImprovement(
        model=acquisition_model,
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
