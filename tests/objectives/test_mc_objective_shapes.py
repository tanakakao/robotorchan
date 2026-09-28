"""Shape contracts from posterior samples through MC objectives."""

import torch
from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler

from robotorchan.models.expressive.deep_gp_posterior import DeepGPPosterior
from robotorchan.models.non_gp.distribution_posterior import GaussianDistributionPosterior
from robotorchan.models.non_gp.posterior import make_ensemble_posterior
from robotorchan.models.standard.multitask import KroneckerMultiTaskGP
from robotorchan.reduction import OutputPCAReducer


def test_gaussian_posterior_preserves_sample_batch_q_output_axes() -> None:
    mean = torch.zeros(2, 3, 1, dtype=torch.double)
    variance = torch.full_like(mean, 0.25)
    posterior = GaussianDistributionPosterior(mean=mean, variance=variance)
    samples = SobolQMCNormalSampler(torch.Size([5, 7]), seed=11)(posterior)
    objective = GenericMCObjective(lambda Y, X=None: Y.squeeze(-1))

    values = objective(samples)

    assert samples.shape == torch.Size([5, 7, 2, 3, 1])
    assert values.shape == torch.Size([5, 7, 2, 3])


def test_ensemble_posterior_preserves_sample_batch_q_output_axes() -> None:
    members = torch.arange(9 * 2 * 3 * 2, dtype=torch.double).reshape(9, 2, 3, 2)
    posterior = make_ensemble_posterior(members)
    samples = IndexSampler(torch.Size([5, 7]), seed=13)(posterior)
    objective = LinearMCObjective(weights=torch.tensor([0.25, 0.75], dtype=torch.double))

    values = objective(samples)

    assert samples.shape == torch.Size([5, 7, 2, 3, 2])
    assert values.shape == torch.Size([5, 7, 2, 3])


def test_deep_gp_posterior_preserves_sample_batch_q_output_axes() -> None:
    trajectories = torch.arange(11 * 2 * 3, dtype=torch.double).reshape(11, 2, 3, 1)
    posterior = DeepGPPosterior(trajectories)
    samples = StochasticSampler(torch.Size([5, 7]))(posterior)
    objective = GenericMCObjective(lambda Y, X=None: Y.squeeze(-1))

    values = objective(samples)

    assert samples.shape == torch.Size([5, 7, 2, 3, 1])
    assert values.shape == torch.Size([5, 7, 2, 3])


def test_kronecker_q_one_keeps_output_axis_distinct_from_q() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (
            torch.sin(3.0 * train_X),
            torch.cos(3.0 * train_X),
        ),
        dim=-1,
    )
    model = KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, rank=1)
    candidate = torch.tensor([[0.4]], dtype=torch.double)
    samples = SobolQMCNormalSampler(torch.Size([5, 7]), seed=17)(model.posterior(candidate))
    objective = LinearMCObjective(weights=torch.tensor([0.4, 0.6], dtype=torch.double))

    values = objective(samples)

    assert samples.shape == torch.Size([5, 7, 1, 2])
    assert values.shape == torch.Size([5, 7, 1])


def test_kronecker_q_many_keeps_output_axis_distinct_from_q() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (
            torch.sin(3.0 * train_X),
            torch.cos(3.0 * train_X),
        ),
        dim=-1,
    )
    model = KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, rank=1)
    candidate = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.double)
    samples = SobolQMCNormalSampler(torch.Size([5, 7]), seed=19)(model.posterior(candidate))
    objective = LinearMCObjective(weights=torch.tensor([0.4, 0.6], dtype=torch.double))

    values = objective(samples)

    assert samples.shape == torch.Size([5, 7, 3, 2])
    assert values.shape == torch.Size([5, 7, 3])


def test_output_reducer_inverse_preserves_sample_batch_q_axes() -> None:
    train_Y = torch.randn(20, 5, dtype=torch.double)
    reducer = OutputPCAReducer(n_components=2).fit(train_Y)
    latent_samples = torch.randn(5, 7, 2, 3, 2, dtype=torch.double)
    restored = reducer.inverse_transform(latent_samples)
    objective = LinearMCObjective(
        weights=torch.tensor([0.1, 0.2, 0.3, 0.15, 0.25], dtype=torch.double)
    )

    values = objective(restored)

    assert restored.shape == torch.Size([5, 7, 2, 3, 5])
    assert values.shape == torch.Size([5, 7, 2, 3])
