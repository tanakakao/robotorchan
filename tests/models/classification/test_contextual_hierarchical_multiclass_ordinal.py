"""Phase 10: contextual and nested hierarchical multiclass/ordinal classifiers."""

import pytest
import torch
from gpytorch.kernels import AdditiveKernel

from robotorchan.acquisition.active_learning.classification import (
    PredictiveEntropy,
    qBALD,
    qPredictiveEntropy,
)
from robotorchan.models.classification import (
    ContextualMulticlassGPClassifier,
    ContextualOrdinalGPClassifier,
    HierarchicalMulticlassGPClassifier,
    HierarchicalOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "model_type",
    [
        ContextualMulticlassGPClassifier,
        ContextualOrdinalGPClassifier,
        HierarchicalMulticlassGPClassifier,
        HierarchicalOrdinalGPClassifier,
    ],
)
def test_contextual_hierarchical_fit_predict_and_candidate(model_type):
    torch.manual_seed(52)
    x = torch.rand(12, 3, dtype=torch.double)
    hierarchical = model_type in (
        HierarchicalMulticlassGPClassifier,
        HierarchicalOrdinalGPClassifier,
    )
    if hierarchical:
        x[:, 1] = (torch.arange(12) // 6).to(torch.double)
        x[:, 2] = (torch.arange(12) // 3).to(torch.double)
        model = model_type(x, torch.arange(12) % 3, 1, 2, num_classes=3)
        assert model.num_groups == 2
        assert model.num_subgroups == 4
    else:
        x[:, 2] = (torch.arange(12) % 2).to(torch.double)
        model = model_type(x, torch.arange(12) % 3, 2, num_classes=3)
        assert model.num_contexts == 2
    kernel = (
        model.class_models[0].model.covar_module
        if isinstance(
            model,
            (ContextualMulticlassGPClassifier, HierarchicalMulticlassGPClassifier),
        )
        else model.model.covar_module
    )
    assert isinstance(kernel, AdditiveKernel)
    assert len(kernel.kernels) == (3 if hierarchical else 2)
    model.fit(epochs=2, lr=0.005)
    pool = x[:8]
    with torch.no_grad():
        probabilities = model.predict_proba(pool)
        posterior = (
            model.probability_posterior(pool, num_samples=16, seed=19)
            if isinstance(
                model,
                (ContextualMulticlassGPClassifier, HierarchicalMulticlassGPClassifier),
            )
            else model.probability_posterior(pool)
        )
        scores = PredictiveEntropy(model)(pool.unsqueeze(-2))
        q_scores = qPredictiveEntropy(model)(pool.reshape(4, 2, 3))
        joint_scores = qBALD(model, num_samples=8)(pool.reshape(4, 2, 3))
    assert probabilities.shape == (8, 3)
    assert posterior.mean.shape == (8, 3)
    assert posterior.variance.shape == (8, 3)
    assert scores.shape == (8,)
    assert q_scores.shape == (4,)
    assert joint_scores.shape == (4,)
    assert torch.isfinite(joint_scores).all()
    torch.testing.assert_close(
        probabilities.sum(-1),
        torch.ones(8, dtype=torch.double),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(posterior.variance).all()
    assert pool[scores.argmax()].shape == (3,)


@pytest.mark.parametrize(
    "model_type",
    [ContextualMulticlassGPClassifier, ContextualOrdinalGPClassifier],
)
def test_contextual_rejects_noninteger_ids_and_invalid_rank(model_type):
    x = torch.rand(8, 2, dtype=torch.double)
    y = torch.arange(8) % 3
    with pytest.raises(ValueError, match="integer"):
        model_type(x, y, context_feature=1, num_classes=3)
    x[:, 1] = (torch.arange(8) % 2).to(torch.double)
    with pytest.raises(ValueError, match="rank"):
        model_type(x, y, context_feature=1, num_classes=3, rank=3)
    with pytest.raises(ValueError, match="non-context"):
        model_type(x[:, 1:], y, context_feature=0, num_classes=3)


@pytest.mark.parametrize(
    "model_type",
    [HierarchicalMulticlassGPClassifier, HierarchicalOrdinalGPClassifier],
)
def test_hierarchy_rejects_shared_subgroups_and_duplicate_features(model_type):
    x = torch.rand(8, 3, dtype=torch.double)
    y = torch.arange(8) % 3
    x[:, 1] = (torch.arange(8) % 2).to(torch.double)
    x[:, 2] = (torch.arange(8) % 2).to(torch.double)
    with pytest.raises(ValueError, match="distinct"):
        model_type(x, y, group_feature=1, subgroup_feature=1, num_classes=3)
    x[:, 2] = 0.0
    with pytest.raises(ValueError, match="exactly one group"):
        model_type(x, y, group_feature=1, subgroup_feature=2, num_classes=3)
