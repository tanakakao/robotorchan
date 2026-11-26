"""Decoupled acquisition composition across semantic output types."""

import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP

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
    MixedBinarySingleTaskGPClassifier,
    MixedMulticlassSingleTaskGPClassifier,
    MixedOrdinalSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.optim import optimize_acqf_decoupled
from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace


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


@pytest.mark.parametrize("kind", ["binary", "multiclass", "ordinal"])
def test_mixed_decoupled_al_and_particle_kg(kind):
    torch.manual_seed(17)
    X = torch.rand(12, 2, dtype=torch.double)
    X[:, 1] = torch.arange(12) % 2
    classes = {
        "binary": MixedBinarySingleTaskGPClassifier,
        "multiclass": MixedMulticlassSingleTaskGPClassifier,
        "ordinal": MixedOrdinalSingleTaskGPClassifier,
    }
    kwargs = {} if kind == "binary" else {"num_classes": 3}
    labels = torch.arange(12) % (2 if kind == "binary" else 3)
    classifier = classes[kind](X, labels, cat_dims=[1], inducing_points=6, **kwargs)
    classifier.fit(epochs=2, lr=0.01)
    regression = MixedSingleTaskGP(X, X[:, :1].sin(), cat_dims=[1])
    regression.eval()
    model = HeterogeneousModel(regression, classifier)
    mask = torch.tensor([[True, False], [False, True]])
    bounds = X.new_tensor([[0.0, 0.0], [1.0, 1.0]])
    space = MixedVariableSpace(bounds, categorical_values={1: (0.0, 1.0)})
    al = create_acquisition(
        "qDecoupledActiveLearning",
        model=model,
        components=[qPosteriorVariance(regression), qPredictiveEntropy(classifier)],
        X_evaluation_mask=mask,
        output_weights=X.new_ones(2),
    )

    def utility(samples):
        probabilities = samples[1]
        rewards = torch.arange(probabilities.shape[-1], dtype=X.dtype, device=X.device)
        return (probabilities * rewards).sum(dim=-1)

    kg = create_acquisition(
        "qDecoupledParticleKnowledgeGradient",
        model=model,
        decision_X=X[:4],
        utility=utility,
        X_evaluation_mask=mask,
        observation_noise=[0.1, None],
        num_particles=16,
        num_fantasies=4,
    )
    for acq in (al, kg):
        points, selected, score = optimize_acqf_decoupled(
            acq,
            bounds,
            mask,
            q=2,
            variable_space=space,
            optimizer="sobol",
            optimizer_options={"num_samples": 8},
            seed=42,
        )
        assert points.shape == (2, 2)
        assert selected.shape == (2, 2)
        assert ((points[:, 1] == 0) | (points[:, 1] == 1)).all()
        assert torch.isfinite(score)
    with pytest.raises(ValueError, match="derivative-free"):
        optimize_acqf_decoupled(kg, bounds, mask)


def test_particle_kg_constant_utility_has_zero_information_value():
    X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    model = HeterogeneousModel(SingleTaskGP(X, X.sin()))
    model.eval()
    acq = create_acquisition(
        "qDecoupledParticleKnowledgeGradient",
        model=model,
        decision_X=X[:3],
        utility=lambda samples: torch.ones_like(samples[0].squeeze(-1)),
        X_evaluation_mask=torch.ones(1, 1, dtype=torch.bool),
        observation_noise=[0.1],
        num_particles=16,
        num_fantasies=4,
    )
    actual = acq(X[3:5].unsqueeze(-2))
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=1e-12, rtol=0)
