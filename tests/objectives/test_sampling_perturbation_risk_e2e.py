"""E2E sampling contracts for input perturbation and risk aggregation."""

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP
from robotorchan.objectives import CVaR, Expectation, MeanVariance
from robotorchan.uncertainty import GaussianPerturbation


def _model() -> SingleTaskGP:
    train_x = torch.rand(16, 2, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 3.0) - 0.2 * train_x[:, 1:2]
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    return model


def _scenario_samples(
    model: SingleTaskGP,
    candidate: torch.Tensor,
    *,
    n_w: int = 8,
) -> torch.Tensor:
    scenarios = GaussianPerturbation(std=0.03).sample(candidate, n_w=n_w)
    posterior = model.posterior(scenarios)
    sampler = SobolQMCNormalSampler(torch.Size([32]), seed=123)
    return sampler(posterior).squeeze(-1)


def test_input_and_posterior_sampling_axes_remain_distinct() -> None:
    model = _model()
    candidate = torch.rand(2, 2, dtype=torch.double)
    samples = _scenario_samples(model, candidate, n_w=8)

    assert samples.shape == torch.Size([32, 2, 8])
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize(
    "risk_measure",
    [
        Expectation(),
        MeanVariance(risk_weight=0.2),
        CVaR(alpha=0.75),
    ],
)
def test_risk_aggregation_reduces_only_scenario_axis(risk_measure) -> None:
    model = _model()
    candidate = torch.rand(2, 2, dtype=torch.double)
    samples = _scenario_samples(model, candidate, n_w=8)

    robust_samples = risk_measure(samples)

    assert robust_samples.shape == torch.Size([32, 2])
    assert torch.isfinite(robust_samples).all()


@pytest.mark.parametrize(
    "risk_measure",
    [
        Expectation(),
        MeanVariance(risk_weight=0.2),
        CVaR(alpha=0.75),
    ],
)
def test_perturbation_sampling_and_risk_preserve_candidate_gradient(risk_measure) -> None:
    model = _model()
    candidate = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
    samples = _scenario_samples(model, candidate, n_w=8)
    robust_value = risk_measure(samples).mean()

    gradient = torch.autograd.grad(robust_value, candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()
