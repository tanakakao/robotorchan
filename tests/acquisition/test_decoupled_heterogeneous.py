"""Decoupled acquisition composition across semantic output types."""

import pytest
import torch
from botorch.models import SingleTaskGP

from robotorchan.acquisition import (
    create_acquisition,
    qBALD,
    qLatentStraddle,
    qMarginUncertainty,
    qPosteriorVariance,
    qPredictiveEntropy,
    qProbabilityVariance,
)
from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.optim import optimize_acqf_decoupled
from robotorchan.optim.constraints.contracts import CandidateConstraints


@pytest.mark.parametrize("kind", ["binary", "multiclass", "ordinal"])
def test_decoupled_heterogeneous_al(kind):
    torch.manual_seed(23)
    X = torch.rand(15, 2, dtype=torch.double)
    if kind == "binary":
        classifier = BinarySingleTaskGPClassifier(X, (X[:, 0] > 0.5).long(), inducing_points=6)
    else:
        cls = (
            MulticlassSingleTaskGPClassifier
            if kind == "multiclass"
            else OrdinalSingleTaskGPClassifier
        )
        classifier = cls(X, torch.arange(15) % 3, num_classes=3, inducing_points=6)
    classifier.fit(epochs=2, lr=0.01)
    regression = SingleTaskGP(X, X[:, :1].sin())
    regression.eval()
    model = HeterogeneousModel(regression, classifier)
    assert model.num_outputs == 2
    mask = torch.tensor([[True, False], [False, True]])
    weights = X.new_tensor([1.0, 0.5])
    candidate = X[:4].reshape(2, 2, 2)
    methods = [
        qPredictiveEntropy(classifier),
        qMarginUncertainty(classifier),
        qProbabilityVariance(classifier, num_samples=8),
        qBALD(classifier, num_samples=8),
    ]
    if kind == "binary":
        methods.append(qLatentStraddle(classifier))
    for method in methods:
        variance = qPosteriorVariance(regression)
        acq = create_acquisition(
            "qDecoupledActiveLearning",
            model=model,
            components=[variance, method],
            X_evaluation_mask=mask,
            output_weights=weights,
        )
        torch.manual_seed(42)
        actual = acq(candidate)
        torch.manual_seed(42)
        expected = variance(candidate[:, :1]) + 0.5 * method(candidate[:, 1:])
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (2,)
        assert torch.isfinite(actual).all()

    acq = create_acquisition(
        "qDecoupledActiveLearning",
        model=model,
        components=[qPosteriorVariance(regression), qPredictiveEntropy(classifier)],
        X_evaluation_mask=mask,
        output_weights=weights,
    )
    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), X.new_tensor([1.0]), 0.25),),
    )
    points, selected, score = optimize_acqf_decoupled(
        acq,
        bounds,
        mask,
        q=2,
        optimizer="sobol",
        optimizer_options={"num_samples": 16},
        seed=42,
        output_costs=X.new_tensor([1.0, 2.0]),
        constraints=constraints,
    )
    assert points.shape == (2, 2)
    assert selected.shape == (2, 2)
    assert (points[:, 0] >= 0.25).all()
    assert torch.isfinite(score)


@pytest.mark.parametrize("kind", ["binary", "multiclass", "ordinal"])
def test_particle_kg(kind):
    torch.manual_seed(23)
    X = torch.rand(12, 2, dtype=torch.double)
    if kind == "binary":
        classifier = BinarySingleTaskGPClassifier(X, (X[:, 0] > 0.5).long(), inducing_points=6)
    else:
        cls = (
            MulticlassSingleTaskGPClassifier
            if kind == "multiclass"
            else OrdinalSingleTaskGPClassifier
        )
        classifier = cls(X, torch.arange(12) % 3, num_classes=3, inducing_points=6)
    classifier.fit(epochs=2, lr=0.01)
    regression = SingleTaskGP(X, X[:, :1].sin())
    regression.eval()
    model = HeterogeneousModel(regression, classifier)
    mask = torch.tensor([[True, False], [False, True]])

    def utility(samples):
        regression_samples, probabilities = samples
        feasible = probabilities[..., -1]
        return feasible * (regression_samples.squeeze(-1) + 1.0) - 1.0

    acq = create_acquisition(
        "qDecoupledParticleKnowledgeGradient",
        model=model,
        decision_X=X[:5],
        utility=utility,
        X_evaluation_mask=mask,
        observation_noise=[0.05, None],
        num_particles=16,
        num_fantasies=4,
        abstain_utility=-1.0,
    )
    values = acq(X[:4].reshape(2, 2, 2))
    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    points, selected, score = optimize_acqf_decoupled(
        acq,
        bounds,
        mask,
        q=1,
        optimizer="sobol",
        optimizer_options={"num_samples": 8},
        seed=42,
        output_costs=X.new_tensor([2.0, 1.0]),
    )
    assert points.shape == (1, 2)
    assert selected.shape == (1, 2)
    assert torch.isfinite(score)
