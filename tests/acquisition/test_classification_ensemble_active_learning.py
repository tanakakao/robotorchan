"""End-to-end tests for ensemble classification active learning."""

import torch

from robotorchan.acquisition import (
    BALD,
    MarginUncertainty,
    PredictiveEntropy,
    ProbabilityVariance,
)
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    HeterogeneousBinaryClassificationEnsemble,
)
from robotorchan.models.classification.binary.non_gp.sklearn import (
    RandomForestBinaryClassifier,
)


def _ensemble(weights: torch.Tensor) -> HeterogeneousBinaryClassificationEnsemble:
    train_X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_Y = (train_X[:, 0] > 0.5).long()
    gp = BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)
    forest = RandomForestBinaryClassifier(
        train_X,
        train_Y,
        n_estimators=8,
        random_state=7,
    )
    forest.fit()
    return HeterogeneousBinaryClassificationEnsemble(
        gp,
        forest,
        weights=weights,
    )


def test_weighted_ensemble_active_learning_matches_probability_posterior() -> None:
    model = _ensemble(torch.tensor([0.8, 0.2], dtype=torch.double))
    X = torch.tensor([[[0.35]], [[0.65]]], dtype=torch.double)
    posterior = model.probability_posterior(X)

    entropy = PredictiveEntropy(model)(X)
    margin = MarginUncertainty(model)(X)
    variance = ProbabilityVariance(model, num_samples=8)(X)
    bald = BALD(model, num_samples=8)(X)

    torch.testing.assert_close(entropy, posterior.predictive_entropy.squeeze(-1))
    top_two = posterior.mean.topk(k=2, dim=-1).values
    expected_margin = 1.0 - (top_two[..., 0] - top_two[..., 1])
    torch.testing.assert_close(margin, expected_margin.squeeze(-1))
    torch.testing.assert_close(variance, posterior.variance.mean(dim=-1).squeeze(-1))
    torch.testing.assert_close(bald, posterior.mutual_information.squeeze(-1))


def test_active_learning_scores_change_with_ensemble_weights() -> None:
    X = torch.tensor([[[0.45]], [[0.55]]], dtype=torch.double)
    balanced = _ensemble(torch.tensor([0.5, 0.5], dtype=torch.double))
    gp_heavy = _ensemble(torch.tensor([0.95, 0.05], dtype=torch.double))

    balanced_bald = BALD(balanced)(X)
    gp_heavy_bald = BALD(gp_heavy)(X)

    assert torch.isfinite(balanced_bald).all()
    assert torch.isfinite(gp_heavy_bald).all()
    assert not torch.allclose(balanced_bald, gp_heavy_bald)


class _ProbabilityOnlyClassifier(torch.nn.Module):
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        positive = X[..., 0].clamp(0.0, 1.0)
        return torch.stack((1.0 - positive, positive), dim=-1)


def test_entropy_and_margin_require_only_predictive_probabilities() -> None:
    model = _ProbabilityOnlyClassifier()
    X = torch.tensor([[[0.25]], [[0.5]]], dtype=torch.double)

    entropy = PredictiveEntropy(model)(X)
    margin = MarginUncertainty(model)(X)

    assert entropy.shape == torch.Size([2])
    assert margin.shape == torch.Size([2])
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(margin).all()
