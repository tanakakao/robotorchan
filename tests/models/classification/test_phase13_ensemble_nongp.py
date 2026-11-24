"""Phase 13 multiclass/ordinal non-GP and probability ensembles."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification import (
    ExtraTreesMulticlassClassifier,
    ExtraTreesOrdinalClassifier,
    GradientBoostingMulticlassClassifier,
    GradientBoostingOrdinalClassifier,
    HistGradientBoostingMulticlassClassifier,
    HistGradientBoostingOrdinalClassifier,
    MulticlassClassificationEnsemble,
    OrdinalClassificationEnsemble,
    RandomForestMulticlassClassifier,
    RandomForestOrdinalClassifier,
)


@pytest.mark.parametrize(
    "model_class",
    [
        RandomForestMulticlassClassifier,
        ExtraTreesMulticlassClassifier,
        GradientBoostingMulticlassClassifier,
        HistGradientBoostingMulticlassClassifier,
        RandomForestOrdinalClassifier,
        ExtraTreesOrdinalClassifier,
        GradientBoostingOrdinalClassifier,
        HistGradientBoostingOrdinalClassifier,
    ],
)
def test_nongp_fit_probabilities_and_candidate_proposal(model_class):
    pytest.importorskip("sklearn")
    torch.manual_seed(33)
    x = torch.rand(24, 2, dtype=torch.double)
    y = (torch.arange(24) % 3).long()
    model = model_class(x, y, num_classes=3, random_state=7)
    assert not model.is_fitted
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(x[:2])
    model.fit()
    with torch.no_grad():
        p = model.predict_proba(x[:8])
        scores = PredictiveEntropy(model)(x[:8].unsqueeze(-2))
    assert p.shape == (8, 3)
    assert scores.shape == (8,)
    assert torch.isfinite(scores).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(8, dtype=x.dtype))
    assert x[:8][scores.argmax()].shape == (2,)
    with pytest.raises(NotImplementedError, match="posterior"):
        model.probability_posterior(x[:2])
    with pytest.raises(NotImplementedError, match="latent"):
        model.latent_posterior(x[:2])


@pytest.mark.parametrize(
    "ensemble_class, member_class",
    [
        (MulticlassClassificationEnsemble, RandomForestMulticlassClassifier),
        (OrdinalClassificationEnsemble, RandomForestOrdinalClassifier),
    ],
)
def test_ensemble_weighted_posterior_and_active_learning(ensemble_class, member_class):
    pytest.importorskip("sklearn")
    torch.manual_seed(42)
    x = torch.rand(24, 2, dtype=torch.double)
    y = (torch.arange(24) % 3).long()
    first = member_class(x, y, num_classes=3, n_estimators=7, random_state=11)
    second = member_class(x, y, num_classes=3, n_estimators=9, random_state=17)
    first.fit()
    second.fit()
    ensemble = ensemble_class(
        first, second, weights=torch.tensor([0.3, 0.7], dtype=torch.double)
    )
    with torch.no_grad():
        posterior = ensemble.probability_posterior(x[:5])
        mean = ensemble.predict_proba(x[:5])
        entropy = PredictiveEntropy(ensemble)(x[:5].unsqueeze(-2))
    expected = 0.3 * first.predict_proba(x[:5]) + 0.7 * second.predict_proba(x[:5])
    torch.testing.assert_close(mean, expected)
    assert posterior.mean.shape == posterior.variance.shape == (5, 3)
    assert posterior.num_members == 2
    assert entropy.shape == (5,)
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(ensemble.mutual_information(x[:5])).all()
    assert x[:5][entropy.argmax()].shape == (2,)
    with pytest.raises(NotImplementedError, match="latent"):
        ensemble.latent_posterior(x[:5])


def test_ensemble_rejects_different_class_semantics():
    pytest.importorskip("sklearn")
    x = torch.rand(9, 2, dtype=torch.double)
    y = torch.arange(9) % 3
    first = RandomForestMulticlassClassifier(x, y, num_classes=3)
    second = RandomForestMulticlassClassifier(x, y, num_classes=4)
    with pytest.raises(ValueError, match="same num_classes"):
        MulticlassClassificationEnsemble(first, second)


def test_nongp_restores_missing_class_probability_axis():
    pytest.importorskip("sklearn")
    x = torch.rand(12, 2, dtype=torch.double)
    y = torch.tensor([0, 1] * 6)
    model = RandomForestMulticlassClassifier(x, y, num_classes=3, n_estimators=8)
    model.fit()
    p = model.predict_proba(x[:4])
    assert p.shape == (4, 3)
    torch.testing.assert_close(p[:, 2], torch.zeros(4, dtype=torch.double))
