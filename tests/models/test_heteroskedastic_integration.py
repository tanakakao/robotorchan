"""BoTorch integration tests for the iterative heteroskedastic GP."""

import torch
from botorch.acquisition import qUpperConfidenceBound

from robotorchan.models import HeteroskedasticSingleTaskGP
from robotorchan.objectives import CVaR
from robotorchan.uncertainty import GaussianPerturbation


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(19)
    X = torch.linspace(0.0, 1.0, 14, dtype=torch.double).unsqueeze(-1)
    scale = 0.02 + 0.12 * X
    Y = torch.sin(2 * torch.pi * X) + scale * torch.randn_like(X)
    return X, Y


def _fitted_model() -> HeteroskedasticSingleTaskGP:
    X, Y = _data()
    model = HeteroskedasticSingleTaskGP(X, Y)
    return model.fit_heteroskedastic(iterations=1)


def test_q_ucb_accepts_heteroskedastic_model() -> None:
    model = _fitted_model()
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    value = qUpperConfidenceBound(model=model, beta=0.2)(X)
    assert value.shape == torch.Size([2])
    assert torch.isfinite(value).all()


def test_noise_prediction_supports_batch_and_q_dimensions() -> None:
    model = _fitted_model()
    X = torch.tensor([[[0.2], [0.4]], [[0.6], [0.8]]], dtype=torch.double)
    noise = model.predicted_noise(X)
    assert noise.shape == torch.Size([2, 2, 1])
    assert torch.isfinite(noise).all()
    assert torch.all(noise >= model.noise_floor)


def test_scenario_risk_composition() -> None:
    model = _fitted_model()
    X = torch.tensor([[0.3], [0.7]], dtype=torch.double)
    scenarios = GaussianPerturbation(std=0.01).sample(X, n_w=4)
    flat = scenarios.reshape(-1, 1)
    values = model.posterior(flat).mean.reshape(2, 4)
    robust = CVaR(alpha=0.75)(values)
    assert robust.shape == torch.Size([2])
    assert torch.isfinite(robust).all()


def test_dtype_conversion_moves_noise_model() -> None:
    model = _fitted_model().float()
    X = torch.tensor([[0.5]], dtype=torch.float)
    assert model.predicted_noise(X).dtype == torch.float
