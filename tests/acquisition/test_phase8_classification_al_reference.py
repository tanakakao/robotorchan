"""Phase 8 independent classification AL score references."""

import torch

from robotorchan.acquisition import LatentStraddle, MarginUncertainty, PredictiveEntropy
from robotorchan.models.classification import BinarySingleTaskGPClassifier


def test_binary_entropy_margin_and_latent_straddle_reference() -> None:
    """Verify label entropy, top-two margin and latent boundary arithmetic."""
    train_x = torch.tensor([[0.0], [0.3], [0.7], [1.0]], dtype=torch.float64)
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    candidates = torch.tensor([[[0.2]], [[0.5]], [[0.8]]], dtype=torch.float64)
    beta = 1.5

    with torch.no_grad():
        probabilities = model.predict_proba(candidates)
        latent = model.latent_posterior(candidates)
        expected_entropy = -torch.special.xlogy(
            probabilities, probabilities
        ).sum(dim=-1).squeeze(-1)
        sorted_probabilities = probabilities.sort(dim=-1, descending=True).values
        expected_margin = (
            1.0 - sorted_probabilities[..., 0] + sorted_probabilities[..., 1]
        ).squeeze(-1)
        expected_latent = (
            beta * latent.variance.clamp_min(0.0).sqrt() - latent.mean.abs()
        ).squeeze(-1).squeeze(-1)

        actual_entropy = PredictiveEntropy(model)(candidates)
        actual_margin = MarginUncertainty(model)(candidates)
        actual_latent = LatentStraddle(model, beta=beta)(candidates)

    for actual, expected in (
        (actual_entropy, expected_entropy),
        (actual_margin, expected_margin),
        (actual_latent, expected_latent),
    ):
        assert actual.shape == (3,)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    assert torch.all((actual_entropy >= 0.0) & (actual_entropy <= torch.log(torch.tensor(2.0))))
    assert torch.all((actual_margin >= 0.0) & (actual_margin <= 1.0))
