"""DeepGP multiclass and ordinal classification executable regression tests."""

import pytest
import torch
from botorch.posteriors.ensemble import EnsemblePosterior

from robotorchan.acquisition.active_learning.classification import (
    PredictiveEntropy,
    qBALD,
    qPredictiveEntropy,
)
from robotorchan.models.classification import (
    MulticlassSingleTaskDeepGPClassifier,
    OrdinalSingleTaskDeepGPClassifier,
)


@pytest.mark.parametrize(
    "model_type",
    [MulticlassSingleTaskDeepGPClassifier, OrdinalSingleTaskDeepGPClassifier],
)
def test_deepgp_classification_fit_posterior_and_candidate(model_type):
    torch.manual_seed(71)
    x = torch.rand(12, 2, dtype=torch.double)
    y = torch.bucketize(x[:, 0], torch.tensor([0.33, 0.67], dtype=torch.double))
    model = model_type(
        x,
        y,
        num_classes=3,
        hidden_dims=(2,),
        num_inducing=5,
        posterior_samples=8,
    )
    model.fit(epochs=2, lr=0.005, num_likelihood_samples=4)
    candidates = torch.rand(5, 2, dtype=torch.double)
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(candidates, num_samples=8)
        probabilities = model.predict_proba(candidates, num_samples=8)
        probability_posterior = model.probability_posterior(candidates, num_samples=8)
        samples = model.sample_class_probabilities(
            candidates, sample_shape=torch.Size([4]), num_samples=8
        )
        score = PredictiveEntropy(model)(candidates.unsqueeze(-2))
        qscore = qPredictiveEntropy(model)(candidates[:4].reshape(2, 2, 2))
        joint_score = qBALD(model, num_samples=4)(candidates[:4].reshape(2, 2, 2))
    assert posterior.mean.shape == (5, model.num_outputs)
    assert probabilities.shape == (5, 3)
    assert samples.shape == (4, 5, 3)
    assert probability_posterior.mean.shape == (5, 3)
    assert probability_posterior.variance.shape == (5, 3)
    assert score.shape == (5,)
    assert qscore.shape == (2,)
    assert joint_score.shape == (2,)
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(joint_score).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(5, dtype=torch.double),
        atol=1e-5,
        rtol=1e-5,
    )
    assert candidates[score.argmax()].shape == (2,)
    if model_type is MulticlassSingleTaskDeepGPClassifier:
        assert isinstance(posterior, EnsemblePosterior)


@pytest.mark.parametrize(
    "model_type",
    [MulticlassSingleTaskDeepGPClassifier, OrdinalSingleTaskDeepGPClassifier],
)
def test_deepgp_requires_valid_class_labels(model_type):
    x = torch.rand(6, 2, dtype=torch.double)
    with pytest.raises(ValueError):
        model_type(x, torch.tensor([0, 1, 2, 3, 0, 1]), num_classes=3)


def test_deepgp_multiclass_rejects_invalid_training_controls():
    x = torch.rand(6, 2, dtype=torch.double)
    y = torch.arange(6) % 3
    model = MulticlassSingleTaskDeepGPClassifier(
        x, y, num_classes=3, hidden_dims=(2,), num_inducing=4
    )
    with pytest.raises(ValueError, match="epochs"):
        model.fit(epochs=0)
