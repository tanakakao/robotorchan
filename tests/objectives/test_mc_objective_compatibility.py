"""MC Objective contracts across representative posterior families."""

import torch
from botorch.acquisition.objective import (
    GenericMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
)
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler

from robotorchan.models.standard.multitask import KroneckerMultiTaskGP
from robotorchan.models.non_gp.posterior import make_ensemble_posterior
from robotorchan.models.expressive.deep_gp_posterior import DeepGPPosterior


def _assert_linear_objective_contract(samples: torch.Tensor) -> None:
    weights = torch.tensor([0.25, 0.75], dtype=samples.dtype, device=samples.device)
    objective = LinearMCObjective(weights=weights)

    values = objective(samples)

    expected = (samples * weights).sum(dim=-1)
    torch.testing.assert_close(values, expected)
    assert values.shape == samples.shape[:-1]
    assert values.dtype == samples.dtype
    assert values.device == samples.device


def test_kronecker_samples_support_native_linear_mc_objective() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (
            torch.sin(3.0 * train_X),
            0.7 * torch.sin(3.0 * train_X) + 0.3 * train_X,
        ),
        dim=-1,
    )
    model = KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, rank=1)
    candidate = torch.tensor([[0.25], [0.75]], dtype=torch.double, requires_grad=True)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=123)(
        model.posterior(candidate)
    )

    _assert_linear_objective_contract(samples)
    objective = LinearMCObjective(
        weights=torch.tensor([0.25, 0.75], dtype=torch.double)
    )
    gradient = torch.autograd.grad(objective(samples).mean(), candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()


def test_ensemble_samples_support_native_generic_mc_objective() -> None:
    values = torch.arange(9 * 3 * 2, dtype=torch.double).reshape(9, 3, 2)
    posterior = make_ensemble_posterior(values)
    samples = IndexSampler(torch.Size([16]), seed=456)(posterior)
    objective = GenericMCObjective(
        lambda Y, X=None: Y[..., 0].square() + 0.5 * Y[..., 1]
    )

    result = objective(samples)

    expected = samples[..., 0].square() + 0.5 * samples[..., 1]
    torch.testing.assert_close(result, expected)
    assert result.shape == samples.shape[:-1]


def test_deep_gp_samples_support_native_linear_mc_objective() -> None:
    trajectories = torch.arange(11 * 3 * 2, dtype=torch.double).reshape(11, 3, 2)
    posterior = DeepGPPosterior(trajectories)
    samples = StochasticSampler(torch.Size([16]))(posterior)

    _assert_linear_objective_contract(samples)


def test_identity_mc_objective_preserves_single_output_samples() -> None:
    samples = torch.randn(16, 3, 1, dtype=torch.double)
    objective = IdentityMCObjective()

    values = objective(samples)

    torch.testing.assert_close(values, samples.squeeze(-1))
    assert values.shape == torch.Size([16, 3])
