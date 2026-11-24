"""Phase 13 multiclass/ordinal non-GP and probability ensembles."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification import (
    BootstrapGradientBoostingMulticlassClassifier,
    BootstrapGradientBoostingOrdinalClassifier,
    ExtraTreesMulticlassClassifier,
    ExtraTreesOrdinalClassifier,
    GradientBoostingMulticlassClassifier,
    GradientBoostingOrdinalClassifier,
    HistGradientBoostingMulticlassClassifier,
    HistGradientBoostingOrdinalClassifier,
    MulticlassClassificationEnsemble,
    MulticlassSingleTaskGPClassifier,
    OrdinalClassificationEnsemble,
    OrdinalSingleTaskGPClassifier,
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
    ensemble = ensemble_class(first, second, weights=torch.tensor([0.3, 0.7], dtype=torch.double))
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


@pytest.mark.parametrize(
    "ensemble_class, member_class, gp_class",
    [
        (
            MulticlassClassificationEnsemble,
            RandomForestMulticlassClassifier,
            MulticlassSingleTaskGPClassifier,
        ),
        (
            OrdinalClassificationEnsemble,
            RandomForestOrdinalClassifier,
            OrdinalSingleTaskGPClassifier,
        ),
    ],
)
def test_hybrid_gp_non_gp_ensemble_candidate_acquisition(ensemble_class, member_class, gp_class):
    pytest.importorskip("sklearn")
    torch.manual_seed(13)
    x = torch.linspace(0, 1, 15, dtype=torch.double).unsqueeze(-1)
    y = torch.bucketize(x[:, 0], torch.tensor([0.33, 0.66], dtype=torch.double))
    nongp = member_class(x, y, num_classes=3, n_estimators=8, random_state=9)
    nongp.fit()
    gp = gp_class(x, y, num_classes=3)
    gp.fit(epochs=2, lr=0.005)
    ensemble = ensemble_class(nongp, gp, weights=torch.tensor([0.5, 0.5], dtype=torch.double))
    candidates = x[:6]
    with torch.no_grad():
        probabilities = ensemble.predict_proba(candidates)
        posterior = ensemble.probability_posterior(candidates)
        scores = PredictiveEntropy(ensemble)(candidates.unsqueeze(-2))
    assert probabilities.shape == (6, 3)
    assert posterior.variance.shape == (6, 3)
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()
    assert candidates[scores.argmax()].shape == (1,)


@pytest.mark.parametrize(
    "model_class",
    [BootstrapGradientBoostingMulticlassClassifier, BootstrapGradientBoostingOrdinalClassifier],
)
def test_bootstrap_ensemble_fit_posterior_and_candidate(model_class):
    pytest.importorskip("sklearn")
    torch.manual_seed(20)
    x = torch.linspace(0, 1, 24, dtype=torch.double).unsqueeze(-1)
    y = torch.arange(24) % 3
    ensemble = model_class(x, y, num_classes=3, n_members=3, random_state=8)
    ensemble.fit()
    with torch.no_grad():
        posterior = ensemble.probability_posterior(x[:6])
        scores = PredictiveEntropy(ensemble)(x[:6].unsqueeze(-2))
    assert posterior.num_members == 3
    assert posterior.mean.shape == posterior.variance.shape == (6, 3)
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()
    with pytest.raises(ValueError, match="at least two"):
        model_class(x, y, num_classes=3, n_members=1)
