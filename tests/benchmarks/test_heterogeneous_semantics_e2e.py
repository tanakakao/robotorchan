"""Phase 5: numerical semantic validation on the fixed heterogeneous benchmark."""

import pytest
import torch

from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import (
    ClassificationConstraint,
    ConstraintDirection,
    ContinuousConstraint,
    ObjectiveDirection,
    ProblemSemantics,
    RegressionObjective,
)


def _heterogeneous_model(*, named: bool) -> HeterogeneousModel:
    generator = torch.Generator().manual_seed(505)
    train_X = torch.rand(20, 3, generator=generator, dtype=torch.double)
    observed = observe(train_X, seed=505)
    models = (
        SingleTaskGP(train_X, observed.strength.unsqueeze(-1)),
        SingleTaskGP(train_X, observed.conductivity.unsqueeze(-1)),
        BinarySingleTaskGPClassifier(train_X, observed.passed, inducing_points=8),
    )
    if named:
        return HeterogeneousModel(*models, output_names=["strength", "conductivity", "pass"])
    return HeterogeneousModel(*models)


@pytest.mark.parametrize("named", [False, True])
def test_semantic_output_resolution_and_feasibility(named: bool) -> None:
    """Resolve two objectives and one classification constraint by name or index."""
    model = _heterogeneous_model(named=named)
    strength = "strength" if named else 0
    conductivity = "conductivity" if named else 1
    passed = "pass" if named else 2
    semantics = ProblemSemantics(
        objectives=(
            RegressionObjective(strength, ObjectiveDirection.MAXIMIZE),
            RegressionObjective(conductivity, ObjectiveDirection.MAXIMIZE),
        ),
        constraints=(ClassificationConstraint(passed, feasible_class=1),),
    )
    semantics.validate(model)
    assert semantics.resolve_objective_outputs(model) == (0, 1)
    assert semantics.resolve_constraint_outputs(model) == (2,)
    assert semantics.resolve_constraint_owners(model) == ((2, 0),)

    X = torch.tensor([[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]], dtype=torch.double)
    representations = semantics.feasibility_representations(model)
    assert len(representations) == 1
    probabilities = model.entry_predict_proba(2, X)
    assert probabilities.shape == (2, 2)
    assert torch.isfinite(probabilities).all()


def test_objective_direction_and_continuous_constraint_residual() -> None:
    """Check maximization convention and BoTorch <= 0 residual numerically."""
    model = _heterogeneous_model(named=False)
    samples = torch.tensor([[[0.7], [1.2]]], dtype=torch.double)
    maximize = RegressionObjective(0, ObjectiveDirection.MAXIMIZE).to_botorch(model)
    minimize = RegressionObjective(0, ObjectiveDirection.MINIMIZE).to_botorch(model)
    torch.testing.assert_close(maximize(samples), samples[..., 0])
    torch.testing.assert_close(minimize(samples), -samples[..., 0])

    constraint = ContinuousConstraint(
        output=0,
        threshold=1.0,
        direction=ConstraintDirection.GREATER_THAN_OR_EQUAL,
    )
    residual = constraint.to_botorch(model)(samples)
    torch.testing.assert_close(residual, torch.tensor([[0.3, -0.2]], dtype=torch.double))


def test_classification_constraint_probability_matches_classifier() -> None:
    """Pass feasibility must use P(Pass), not the latent GP mean."""
    model = _heterogeneous_model(named=True)
    X = torch.tensor(
        [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
        dtype=torch.double,
    )
    constraint = ClassificationConstraint("pass", feasible_class=1)
    model.eval()
    with torch.no_grad():
        expected = model.entry_predict_proba(2, X)[..., 1]
        actual = constraint.to_probability_of_feasibility(model)(X)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2,)
    assert torch.isfinite(actual).all()


def test_probability_threshold_residual_sign() -> None:
    """Thresholded classification feasibility is threshold minus P(Pass)."""
    model = _heterogeneous_model(named=True)
    X = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.double)
    constraint = ClassificationConstraint("pass", feasible_class=1, probability_threshold=0.6)
    with torch.no_grad():
        probability = model.entry_predict_proba(2, X)[..., 1]
        residual = constraint.to_probability_constraint(model)(X)
    torch.testing.assert_close(residual, 0.6 - probability)


def test_benchmark_truth_is_distinct_from_model_semantics() -> None:
    """Benchmark truth is evaluation data, not a semantic model prediction."""
    X = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.double)
    strength, conductivity, probability = evaluate_truth(X)
    assert strength.shape == conductivity.shape == probability.shape == (1,)
    assert torch.all((probability >= 0) & (probability <= 1))
