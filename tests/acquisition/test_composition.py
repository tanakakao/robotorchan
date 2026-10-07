import pytest
import torch
from botorch.acquisition.objective import GenericMCObjective

from robotorchan.acquisition.composition import (
    make_botorch_objective_bridge,
    resolve_acquisition_composition,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard import SingleTaskGP
from robotorchan.semantics import (
    ClassificationConstraint,
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
