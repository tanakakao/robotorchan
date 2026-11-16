"""Tests for the classification ensemble posterior contract."""

import pytest
import torch
from botorch.posteriors import Posterior
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.sampling.get_sampler import get_sampler
from botorch.sampling.index_sampler import IndexSampler

from robotorchan.models.classification.posterior import (
    ClassificationEnsemblePosterior,
    make_classification_ensemble_posterior,
)


def _probabilities() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.6, 0.4], [0.2, 0.8]],
            [[0.7, 0.3], [0.4, 0.6]],
        ],
        dtype=torch.double,
    )


def test_ensemble_posterior_moments_and_information() -> None:
    probabilities = _probabilities()
    posterior = make_classification_ensemble_posterior(probabilities)

    torch.testing.assert_close(posterior.mean, probabilities.mean(dim=0))
    torch.testing.assert_close(
        posterior.variance,
        probabilities.var(dim=0, unbiased=False),
    )
    assert posterior.predictive_entropy.shape == (2,)
    assert posterior.expected_class_entropy.shape == (2,)
    assert posterior.mutual_information.shape == (2,)
    assert torch.all(posterior.mutual_information >= 0.0)


def test_ensemble_posterior_sampling_preserves_probability_simplex() -> None:
    posterior = ClassificationEnsemblePosterior(_probabilities())
    samples = posterior.rsample(torch.Size([4, 5]))

    assert samples.shape == (4, 5, 2, 2)
    torch.testing.assert_close(
        samples.sum(dim=-1),
        torch.ones(4, 5, 2, dtype=torch.double),
    )


@pytest.mark.parametrize(
    "probabilities, message",
    [
        (torch.tensor([[0, 1], [1, 0]]), "floating-point"),
        (torch.tensor([[0.2, 0.2], [0.5, 0.5]]), "sum to one"),
        (torch.tensor([[1.2, -0.2], [0.5, 0.5]]), r"\[0, 1\]"),
        (torch.tensor([[0.5, 0.5]]), "at least two members"),
    ],
)
def test_ensemble_posterior_rejects_invalid_probabilities(
    probabilities: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ClassificationEnsemblePosterior(probabilities)


def test_ensemble_posterior_preserves_autograd() -> None:
    logits = torch.tensor(
        [[[-1.0, 1.0]], [[0.5, -0.5]]],
        dtype=torch.double,
        requires_grad=True,
    )
    probabilities = logits.softmax(dim=-1)
    posterior = ClassificationEnsemblePosterior(probabilities)
    posterior.mutual_information.sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_weighted_ensemble_posterior_uses_weights_for_all_moments() -> None:
    probabilities = _probabilities()
    weights = torch.tensor([0.6, 0.3, 0.1], dtype=torch.double)
    posterior = ClassificationEnsemblePosterior(probabilities, weights=weights)
    normalized = weights / weights.sum()
    view = normalized[:, None, None]
    expected_mean = (probabilities * view).sum(dim=0)
    expected_variance = ((probabilities - expected_mean) ** 2 * view).sum(dim=0)

    torch.testing.assert_close(posterior.weights, normalized)
    torch.testing.assert_close(posterior.mean, expected_mean)
    torch.testing.assert_close(posterior.variance, expected_variance)

    safe = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    member_entropy = -torch.special.xlogy(safe, safe).sum(dim=-1)
    torch.testing.assert_close(
        posterior.expected_class_entropy,
        (member_entropy * normalized[:, None]).sum(dim=0),
    )
    torch.testing.assert_close(
        posterior.mutual_information,
        (posterior.predictive_entropy - posterior.expected_class_entropy).clamp_min(0.0),
    )


def test_weighted_ensemble_sampling_follows_degenerate_model_weight() -> None:
    probabilities = _probabilities()
    weights = torch.tensor([0.0, 1.0, 0.0], dtype=torch.double)
    posterior = ClassificationEnsemblePosterior(probabilities, weights=weights)

    samples = posterior.rsample(torch.Size([20]))

    torch.testing.assert_close(
        samples,
        probabilities[1].expand(20, *probabilities.shape[1:]),
    )


@pytest.mark.parametrize(
    "weights, message",
    [
        (torch.tensor([0.5, 0.5], dtype=torch.double), "num_members"),
        (torch.tensor([0.5, -0.1, 0.6], dtype=torch.double), "non-negative"),
        (torch.zeros(3, dtype=torch.double), "positive"),
        (torch.tensor([0.5, float("nan"), 0.5], dtype=torch.double), "finite"),
    ],
)
def test_weighted_ensemble_posterior_rejects_invalid_weights(
    weights: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ClassificationEnsemblePosterior(_probabilities(), weights=weights)


def test_ensemble_posterior_reuses_botorch_sampler_contract() -> None:
    posterior = ClassificationEnsemblePosterior(_probabilities())
    assert isinstance(posterior, Posterior)
    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.device == torch.device("cpu")
    assert posterior.dtype == torch.double
    assert posterior.batch_shape == torch.Size()
    assert posterior.base_sample_shape == torch.Size()
    assert posterior.batch_range == (0, 0)
    assert posterior._extended_shape(torch.Size([2, 3])) == torch.Size([2, 3, 2, 2])

    sampler = get_sampler(posterior, sample_shape=torch.Size([16]), seed=7)
    assert isinstance(sampler, IndexSampler)
    first = sampler(posterior)
    second = sampler(posterior)
    assert first.shape == (16, 2, 2)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.sum(dim=-1), torch.ones_like(first[..., 0]))


def test_ensemble_index_samples_select_complete_members_for_each_q_batch() -> None:
    probabilities = _probabilities().unsqueeze(1).expand(-1, 2, -1, -1).clone()
    probabilities[:, 1] = probabilities[:, 1].flip(dims=[1])
    posterior = ClassificationEnsemblePosterior(probabilities)
    assert posterior.batch_shape == torch.Size([2])
    assert posterior.base_sample_shape == torch.Size([2])
    assert posterior.batch_range == (0, 1)
    assert posterior.mean.shape == (2, 2, 2)
    base_samples = torch.tensor([[0, 2], [1, 0], [2, 1]])
    selected = posterior.rsample_from_base_samples(torch.Size([3]), base_samples)
    expected = torch.stack(
        [
            torch.stack((probabilities[0, 0], probabilities[2, 1])),
            torch.stack((probabilities[1, 0], probabilities[0, 1])),
            torch.stack((probabilities[2, 0], probabilities[1, 1])),
        ]
    )
    assert selected.shape == (3, 2, 2, 2)
    torch.testing.assert_close(selected, expected, atol=0.0, rtol=0.0)
    sampler = get_sampler(posterior, sample_shape=torch.Size([7]), seed=13)
    samples = sampler(posterior)
    assert samples.shape == (7, 2, 2, 2)


def test_ensemble_exact_weighted_marginal_credible_intervals() -> None:
    weights = torch.tensor([0.6, 0.3, 0.1], dtype=torch.double)
    posterior = ClassificationEnsemblePosterior(_probabilities(), weights=weights)
    lower, upper = posterior.credible_interval(0.9)
    torch.testing.assert_close(lower, torch.tensor([[0.6, 0.2], [0.2, 0.6]], dtype=torch.double))
    torch.testing.assert_close(upper, torch.tensor([[0.8, 0.4], [0.4, 0.8]], dtype=torch.double))
    lower_again, upper_again = posterior.credible_interval(0.9, num_samples=8)
    torch.testing.assert_close(lower, lower_again, atol=0.0, rtol=0.0)
    torch.testing.assert_close(upper, upper_again, atol=0.0, rtol=0.0)


def test_ensemble_quantiles_with_single_nonzero_member_and_validation() -> None:
    posterior = ClassificationEnsemblePosterior(
        _probabilities(),
        weights=torch.tensor([0.0, 1.0, 0.0], dtype=torch.double),
    )
    lower, upper = posterior.credible_interval(0.95)
    torch.testing.assert_close(lower, _probabilities()[1])
    torch.testing.assert_close(upper, _probabilities()[1])
    torch.testing.assert_close(posterior.variance, torch.zeros_like(posterior.mean))
    with pytest.raises(ValueError, match="strictly between"):
        posterior.credible_interval(1.0)


def test_ensemble_flat_class_probabilities_gain_explicit_q_axis() -> None:
    probabilities = _probabilities()[:, 0, :]
    posterior = ClassificationEnsemblePosterior(probabilities)
    assert posterior.probabilities.shape == (3, 2)
    assert posterior.mean.shape == (1, 2)
    assert posterior.variance.shape == (1, 2)
    lower, upper = posterior.credible_interval()
    assert lower.shape == (1, 2)
    assert upper.shape == (1, 2)
