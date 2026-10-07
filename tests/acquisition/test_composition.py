import pytest
import torch
from torch import nn

from robotorchan.acquisition.composition import resolve_acquisition_composition
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.semantics import (
    ClassificationConstraint,
    ContinuousConstraint,
    FeasibilityRepresentationKind,
    ProblemSemantics,
    RegressionObjective,
)


from robotorchan.models.standard import SingleTaskGP


@pytest.mark.parametrize("use_names", [False, True])
def test_resolve_acquisition_composition_preserves_output_ownership(use_names: bool) -> None:
    from robotorchan.models.classification.binary.standard.single_task import BinaryGPClassifier

    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_y = torch.tensor([0.0, 1.0, 1.0], dtype=torch.double)
    classifier = BinaryGPClassifier(train_x, train_y)

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

    assert [(item.output_index, item.entry_index, item.local_output_index) for item in plan.objectives] == [
        (0, 0, 0)
    ]
    assert [
        (item.output_index, item.entry_index, item.local_output_index) for item in plan.feasibility
    ] == [(1, 1, 0), (2, 2, 0)]
    assert [item.representation.kind for item in plan.feasibility] == [
        FeasibilityRepresentationKind.SAMPLE_RESIDUAL,
        FeasibilityRepresentationKind.PROBABILITY_OF_FEASIBILITY,
    ]
