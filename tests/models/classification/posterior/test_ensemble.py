"""Tests for the classification ensemble posterior contract."""

import pytest
import torch

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
