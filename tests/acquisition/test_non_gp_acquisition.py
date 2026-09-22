import pytest
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler

pytest.importorskip("sklearn")

from robotorchan.acquisition import make_non_gp_acquisition, validate_non_gp_acquisition
from robotorchan.models import ExtraTreesSurrogate, RandomForestSurrogate, SingleTaskGP


@pytest.mark.parametrize("model_class", [RandomForestSurrogate, ExtraTreesSurrogate])
def test_tree_surrogates_support_mc_acquisition(model_class) -> None:
    train_X = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    model = model_class(train_X, train_Y, n_estimators=8, random_state=0)
    model.fit()

    acqf = make_non_gp_acquisition(
        model,
        qExpectedImprovement,
        best_f=train_Y.max(),
        sampler=IndexSampler(sample_shape=torch.Size([8])),
    )
    validate_non_gp_acquisition(model, acqf)

    value = acqf(torch.tensor([[[0.25], [0.75]]], dtype=torch.double))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_non_gp_acquisition_rejects_analytic_gaussian_acquisition() -> None:
    train_X = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    model = RandomForestSurrogate(train_X, train_Y, n_estimators=8, random_state=1)
    model.fit()
    acqf = ExpectedImprovement(model=model, best_f=train_Y.max())

    with pytest.raises(TypeError, match="Monte Carlo"):
        validate_non_gp_acquisition(model, acqf)


def test_non_gp_acquisition_validator_rejects_gp_model() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 4.0)
    model = SingleTaskGP(train_X, train_Y)
    acqf = qExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )

    with pytest.raises(TypeError, match="non-GP"):
        validate_non_gp_acquisition(model, acqf)


def test_non_gp_acquisition_rejects_gaussian_sampler() -> None:
    train_X = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    model = RandomForestSurrogate(train_X, train_Y, n_estimators=8, random_state=2)
    model.fit()
    acqf = qExpectedImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )

    with pytest.raises(TypeError, match="IndexSampler"):
        validate_non_gp_acquisition(model, acqf)
