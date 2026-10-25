"""Phase 8 deterministic posterior class-probability sample references."""

from types import MethodType

import torch

from robotorchan.acquisition import BALD, ProbabilityVariance
from robotorchan.models.classification import BinarySingleTaskGPClassifier


def test_bald_and_probability_variance_match_sample_moment_references() -> None:
    """Check epistemic probability spread and entropy difference independently."""
    train_x = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float64)
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    candidates = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.float64)
    num_samples = 64

    positive_class = torch.tensor(
        [
            [[0.1], [0.7]],
            [[0.2], [0.8]],
            [[0.4], [0.9]],
            [[0.6], [0.6]],
        ],
        dtype=torch.float64,
    )
    positive_class = positive_class.repeat(num_samples // 4, 1, 1)
    samples = torch.stack((1.0 - positive_class, positive_class), dim=-1)

    def fixed_probability_samples(
        self: BinarySingleTaskGPClassifier,
        X: torch.Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        assert X.shape == candidates.shape
        assert sample_shape == torch.Size([num_samples])
        return samples

    model.sample_class_probabilities = MethodType(fixed_probability_samples, model)
    with torch.no_grad():
        sample_mean = samples.mean(dim=0)
        centered = samples - sample_mean
        expected_variance = centered.square().mean(dim=0).mean(dim=-1).squeeze(-1)
        predictive_entropy = -torch.special.xlogy(
            sample_mean, sample_mean
        ).sum(dim=-1)
        conditional_entropy = -torch.special.xlogy(samples, samples).sum(dim=-1)
        expected_bald = (
            predictive_entropy - conditional_entropy.mean(dim=0)
        ).clamp_min(0.0).squeeze(-1)

        actual_variance = ProbabilityVariance(model, num_samples=num_samples)(candidates)
        actual_bald = BALD(model, num_samples=num_samples)(candidates)

    assert samples.shape == (num_samples, 2, 1, 2)
    assert actual_variance.shape == (2,)
    assert actual_bald.shape == (2,)
    torch.testing.assert_close(actual_variance, expected_variance, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual_bald, expected_bald, atol=1e-10, rtol=1e-10)
    assert torch.all(actual_variance >= 0.0)
    assert torch.all(actual_bald >= 0.0)
