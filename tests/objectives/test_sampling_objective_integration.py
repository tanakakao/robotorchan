"""Sampling-to-Objective integration contracts across posterior families."""

import torch
from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler

from robotorchan.models import SingleTaskGP
from robotorchan.models.expressive.deep_gp_posterior import DeepGPPosterior
from robotorchan.models.non_gp.posterior import make_ensemble_posterior
from robotorchan.models.standard.multitask import KroneckerMultiTaskGP


def test_gaussian_posterior_sampling_flows_directly_into_objective() -> None:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(3.0 * train_x)
    model = SingleTaskGP(train_x, train_y)
    candidate = torch.tensor([[0.2], [0.8]], dtype=torch.double, requires_grad=True)
    sampler = SobolQMCNormalSampler(torch.Size([4, 5]), seed=101)
    objective = GenericMCObjective(lambda Y, X=None: Y.squeeze(-1))

    samples = sampler(model.posterior(candidate))
    values = objective(samples, X=candidate)
    gradient = torch.autograd.grad(values.mean(), candidate)[0]

    assert samples.shape == torch.Size([4, 5, 2, 1])
    assert values.shape == torch.Size([4, 5, 2])
    assert torch.isfinite(gradient).all()


def test_kronecker_sampling_flows_directly_into_linear_objective() -> None:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = torch.cat((torch.sin(3.0 * train_x), torch.cos(3.0 * train_x)), dim=-1)
    model = KroneckerMultiTaskGP(train_X=train_x, train_Y=train_y, rank=1)
    candidate = torch.tensor([[0.2], [0.8]], dtype=torch.double, requires_grad=True)
    sampler = SobolQMCNormalSampler(torch.Size([4, 5]), seed=103)
    objective = LinearMCObjective(weights=torch.tensor([0.3, 0.7], dtype=torch.double))

    samples = sampler(model.posterior(candidate))
    values = objective(samples)
    gradient = torch.autograd.grad(values.mean(), candidate)[0]

    assert samples.shape == torch.Size([4, 5, 2, 2])
    assert values.shape == torch.Size([4, 5, 2])
    assert torch.isfinite(gradient).all()


def test_ensemble_sampling_flows_directly_into_objective() -> None:
    members = torch.arange(9 * 3 * 2, dtype=torch.double).reshape(9, 3, 2)
    posterior = make_ensemble_posterior(members)
    sampler = IndexSampler(torch.Size([4, 5]), seed=107)
    objective = GenericMCObjective(lambda Y, X=None: Y[..., 0] - 0.5 * Y[..., 1])

    samples = sampler(posterior)
    values = objective(samples)

    assert samples.shape == torch.Size([4, 5, 3, 2])
    assert values.shape == torch.Size([4, 5, 3])
    torch.testing.assert_close(values, samples[..., 0] - 0.5 * samples[..., 1])


def test_deep_gp_sampling_flows_directly_into_objective() -> None:
    trajectories = torch.arange(11 * 3, dtype=torch.double).reshape(11, 3, 1)
    posterior = DeepGPPosterior(trajectories)
    sampler = StochasticSampler(torch.Size([4, 5]))
    objective = GenericMCObjective(lambda Y, X=None: Y.squeeze(-1))

    samples = sampler(posterior)
    values = objective(samples)

    assert samples.shape == torch.Size([4, 5, 3, 1])
    assert values.shape == torch.Size([4, 5, 3])
    torch.testing.assert_close(values, samples.squeeze(-1))
