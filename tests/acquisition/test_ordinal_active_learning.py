"""Ordinal GP integration with existing classification active-learning acquisitions."""

import pytest
import torch

from robotorchan.acquisition.active_learning.classification import (
    BALD,
    MarginUncertainty,
    PredictiveEntropy,
    ProbabilityVariance,
    qBALD,
    qMarginUncertainty,
    qPredictiveEntropy,
    qProbabilityVariance,
)
from robotorchan.models.classification import (
    MixedOrdinalSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.optim.backends.sampling import optimize_acqf_sampling
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.optim.strategies.sobol import SobolSearchStrategy


@pytest.fixture(params=[False, True], ids=["continuous", "mixed"])
def model(request: pytest.FixtureRequest) -> OrdinalSingleTaskGPClassifier:
    torch.manual_seed(37)
    X = torch.rand(15, 2, dtype=torch.double)
    y = torch.bucketize(X[:, 0], torch.tensor([0.35, 0.7], dtype=torch.double))
    if request.param:
        X[:, 1] = torch.arange(15) % 3
        classifier = MixedOrdinalSingleTaskGPClassifier(
            X, y, num_classes=3, cat_dims=[1], inducing_points=6
        )
    else:
        classifier = OrdinalSingleTaskGPClassifier(
            X, y, num_classes=3, inducing_points=6
        )
    classifier.fit(epochs=2, lr=0.01)
    return classifier


def test_ordinal_q_one_acquisitions(model: OrdinalSingleTaskGPClassifier) -> None:
    candidates = model.raw_train_X[:4].unsqueeze(-2)
    for acquisition in (
        PredictiveEntropy(model),
        MarginUncertainty(model),
        ProbabilityVariance(model, num_samples=8),
        BALD(model, num_samples=8),
    ):
        scores = acquisition(candidates)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()


def test_ordinal_q_batch_acquisitions(model: OrdinalSingleTaskGPClassifier) -> None:
    candidates = model.raw_train_X[:6].reshape(2, 3, 2)
    for acquisition in (
        qPredictiveEntropy(model),
        qMarginUncertainty(model),
        qProbabilityVariance(model, num_samples=8),
        qBALD(model, num_samples=8),
    ):
        scores = acquisition(candidates)
        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()
    with pytest.raises(ValueError, match="num_classes"):
        qBALD(model, num_samples=8)(model.raw_train_X[:6].reshape(1, 6, 2))


def test_ordinal_sobol_candidate_generation(
    model: OrdinalSingleTaskGPClassifier,
) -> None:
    if isinstance(model, MixedOrdinalSingleTaskGPClassifier):
        pytest.skip("Mixed inputs require explicit categorical variable space.")
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    result = SobolSearchStrategy(bounds, num_samples=8, seed=11).optimize(
        qPredictiveEntropy(model), q=2
    )
    assert result.candidates.shape == (2, 2)
    assert torch.isfinite(result.candidates).all()
    assert result.acquisition_value is not None
    assert torch.isfinite(result.acquisition_value)


def test_ordinal_mixed_constrained_sampling(
    model: OrdinalSingleTaskGPClassifier,
) -> None:
    if not isinstance(model, MixedOrdinalSingleTaskGPClassifier):
        pytest.skip("Only mixed inputs use categorical variable space.")
    bounds = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, categorical_values={1: (0.0, 1.0, 2.0)})
    constraints = CandidateConstraints(
        inequality_constraints=(
            (torch.tensor([0]), torch.tensor([1.0], dtype=torch.double), 0.25),
        ),
    )
    candidates, value = optimize_acqf_sampling(
        qPredictiveEntropy(model),
        bounds,
        q=2,
        num_samples=32,
        method="sobol",
        seed=7,
        variable_space=space,
        constraints=constraints,
    )
    assert candidates.shape == (2, 2)
    assert (candidates[:, 0] >= 0.25).all()
    assert torch.isin(candidates[:, 1], torch.tensor([0.0, 1.0, 2.0])).all()
    assert torch.isfinite(value)
