import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective

from robotorchan.acquisition.composition import (
    make_botorch_objective_bridge,
    make_classification_feasibility_bridge,
    make_continuous_constraint_bridge,
    make_deterministic_pof_acquisition,
    resolve_acquisition_composition,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard import SingleTaskGP
from robotorchan.semantics import (
    ClassificationConstraint,
    ConstraintDirection,
    ContinuousConstraint,
    FeasibilityRepresentationKind,
    ObjectiveDirection,
    ProblemSemantics,
    RegressionObjective,
)


@pytest.mark.parametrize("use_names", [False, True])
def test_resolve_acquisition_composition_preserves_output_ownership(use_names: bool) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_y = torch.tensor([0.0, 1.0, 1.0], dtype=torch.double)
    classifier = BinarySingleTaskGPClassifier(train_x, train_y)

    regression_y = torch.sin(train_x)
    model = HeterogeneousModel(
        SingleTaskGP(train_x, regression_y),
        SingleTaskGP(train_x, regression_y),
        classifier,
        output_names=("objective", "cost", "pass") if use_names else None,
    )
    objective = "objective" if use_names else 0
    cost = "cost" if use_names else 1
    passed = "pass" if use_names else 2
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(objective),),
        constraints=(
            ContinuousConstraint(cost, threshold=1.0),
            ClassificationConstraint(passed, feasible_class=1),
        ),
    )

    plan = resolve_acquisition_composition(model, semantics)

    assert [
        (item.output_index, item.entry_index, item.local_output_index) for item in plan.objectives
    ] == [(0, 0, 0)]
    assert [
        (item.output_index, item.entry_index, item.local_output_index) for item in plan.feasibility
    ] == [(1, 1, 0), (2, 2, 0)]
    assert [item.representation.kind for item in plan.feasibility] == [
        FeasibilityRepresentationKind.SAMPLE_RESIDUAL,
        FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
    ]


@pytest.mark.parametrize("direction_sign", [1.0, -1.0])
def test_botorch_objective_bridge_preserves_native_model_and_direction(
    direction_sign: float,
) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_y = torch.sin(train_x)
    regression = SingleTaskGP(train_x, train_y)
    model = HeterogeneousModel(regression)
    direction = ObjectiveDirection.MAXIMIZE if direction_sign > 0 else ObjectiveDirection.MINIMIZE
    semantics = ProblemSemantics(objectives=(RegressionObjective(0, direction=direction),))

    binding = resolve_acquisition_composition(model, semantics).objectives[0]
    bridge = make_botorch_objective_bridge(model, binding)
    samples = torch.tensor([[[[2.0]]]], dtype=torch.double)

    assert bridge.model is regression
    assert isinstance(bridge.objective, GenericMCObjective)
    assert torch.equal(bridge.objective(samples), direction_sign * samples[..., 0])


class _ConstantAcquisition(AcquisitionFunction):
    def __init__(self, model: SingleTaskGP, value: float) -> None:
        super().__init__(model=model)
        self.value = value

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.new_full(X.shape[:-2], self.value)


def test_deterministic_pof_bridge_reuses_probability_representation() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    classifier = BinarySingleTaskGPClassifier(
        train_x,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.double),
    )
    model = HeterogeneousModel(regression, classifier)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ClassificationConstraint(1, feasible_class=1),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]
    objective_acquisition = _ConstantAcquisition(regression, 2.0)

    acquisition = make_deterministic_pof_acquisition(objective_acquisition, binding)
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    expected = 2.0 * binding.representation.probability(X).squeeze(-1)

    assert torch.allclose(acquisition(X), expected)


def test_deterministic_pof_bridge_rejects_sample_residual() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(0, threshold=1.0),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]

    with pytest.raises(TypeError, match="ProbabilityOfFeasibility"):
        make_deterministic_pof_acquisition(_ConstantAcquisition(regression, 2.0), binding)


@pytest.mark.parametrize("use_names", [False, True])
def test_classification_feasibility_bridge_preserves_semantic_pof(use_names: bool) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    classifier = BinarySingleTaskGPClassifier(
        train_x,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.double),
    )
    model = HeterogeneousModel(
        regression,
        classifier,
        output_names=("objective", "pass") if use_names else None,
    )
    output = "pass" if use_names else 1
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ClassificationConstraint(output, feasible_class=1),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]

    probability = make_classification_feasibility_bridge(model, binding)
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)

    assert probability is binding.representation.probability
    assert probability.model is classifier
    assert torch.equal(probability(X), classifier.predict_proba(X)[..., 1])


def test_classification_feasibility_bridge_rejects_non_probability_representation() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(0, threshold=1.0),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]

    with pytest.raises(TypeError, match="ProbabilityOfFeasibility"):
        make_classification_feasibility_bridge(model, binding)


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (ConstraintDirection.LESS_THAN_OR_EQUAL, torch.tensor([-1.0, 1.0])),
        (ConstraintDirection.GREATER_THAN_OR_EQUAL, torch.tensor([1.0, -1.0])),
    ],
)
def test_continuous_constraint_bridge_preserves_sample_residual(
    direction: ConstraintDirection,
    expected: torch.Tensor,
) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(0, threshold=2.0, direction=direction),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]
    constraint = make_continuous_constraint_bridge(model, binding)
    samples = torch.tensor([[[1.0], [3.0]]], dtype=torch.double)

    assert torch.equal(constraint(samples), expected.to(dtype=torch.double).unsqueeze(0))


def test_continuous_constraint_bridge_rejects_probability_representation() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    classifier = BinarySingleTaskGPClassifier(
        train_x,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.double),
    )
    model = HeterogeneousModel(regression, classifier)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ClassificationConstraint(1, feasible_class=1),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]

    with pytest.raises(TypeError, match="SampleResidualFeasibility"):
        make_continuous_constraint_bridge(model, binding)
