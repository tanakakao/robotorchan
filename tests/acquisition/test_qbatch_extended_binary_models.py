"""q-batch active-learning integration for extended binary GP classifiers."""

import pytest
import torch

from robotorchan.acquisition import (
    qBALD,
    qLatentStraddle,
    qMarginUncertainty,
    qPredictiveEntropy,
    qProbabilityVariance,
)
from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.models.classification import get_classification_model_class


@pytest.mark.parametrize(
    ("model_id", "kwargs", "features"),
    [
        ("binary.multifidelity", {"fidelity_features": [1]}, 2),
        ("binary.mixed_multifidelity", {"cat_dims": [1], "fidelity_features": [2]}, 3),
        ("binary.spectral_mixture", {"num_mixtures": 2}, 1),
        ("binary.mixed_spectral_mixture", {"cat_dims": [1], "num_mixtures": 2}, 2),
        ("binary.contextual", {"context_feature": 1}, 2),
        ("binary.hierarchical", {"group_feature": 1, "subgroup_feature": 2}, 3),
        ("binary.infinite_width_bnn", {"depth": 2}, 1),
        ("binary.mixed_infinite_width_bnn", {"cat_dims": [1], "depth": 2}, 2),
    ],
)
def test_extended_binary_models_support_qbatch_active_learning(model_id, kwargs, features):
    x = torch.linspace(0.05, 0.95, 12, dtype=torch.double)
    if features == 1:
        X = x.unsqueeze(-1)
    elif features == 2:
        X = torch.stack((x, torch.tensor([0.0, 1.0] * 6, dtype=x.dtype)), dim=-1)
    else:
        X = torch.stack(
            (
                x,
                torch.tensor([0.0, 1.0] * 6, dtype=x.dtype),
                torch.tensor([0.0, 2.0, 1.0, 3.0] * 3, dtype=x.dtype),
            ),
            dim=-1,
        )
    Y = (x > 0.5).long()
    model_class = get_classification_model_class(model_id)
    model = model_class(X, Y, inducing_points=6, **kwargs)
    model.eval()
    for q in (1, 2, 3):
        candidates = X[:q].unsqueeze(0)
        for acquisition in (
            qPredictiveEntropy(model),
            qMarginUncertainty(model),
            qProbabilityVariance(model, num_samples=8),
            qLatentStraddle(model),
            qBALD(model, num_samples=8),
        ):
            score = acquisition(candidates)
            assert score.shape == torch.Size([1]), (model_id, q, type(acquisition))
            assert torch.isfinite(score).all(), (model_id, q, type(acquisition))


def test_qbald_limits_and_optimizer_capabilities():
    for q in (1, 2, 8):
        assert check_acquisition_optimizer_compatibility(
            "qBALD", CandidateOptimizer.CONTINUOUS, q=q
        ).compatible
    assert not check_acquisition_optimizer_compatibility(
        "qBALD", CandidateOptimizer.CONTINUOUS, q=9
    ).compatible
