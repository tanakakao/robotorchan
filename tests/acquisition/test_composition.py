import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.classification_constraints import (
    FeasibilityWeightedAcquisition,
    IndependentFeasibilityAggregator,
)
from robotorchan.acquisition.composition import (
    SampleShapeContract,
    make_botorch_objective_bridge,
    make_classification_feasibility_bridge,
    make_continuous_constrained_qei_acquisition,
    make_continuous_constrained_qnei_acquisition,
    make_continuous_constraint_bridge,
    make_deterministic_pof_acquisition,
    make_mixed_constrained_qei_acquisition,
    make_multiple_learned_constrained_qnei_acquisition,
    make_qei_acquisition,
    make_qnei_acquisition,
    make_sample_classification_feasibility_bridge,
    resolve_acquisition_composition,
)
from robotorchan.models.classification.binary.non_gp.sklearn import (
    RandomForestBinaryClassifier,
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


def test_sample_classification_feasibility_preserves_mc_sample_dimension() -> None:
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
    sample_feasibility = make_sample_classification_feasibility_bridge(model, binding)
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    torch.manual_seed(7)
    expected = classifier.sample_class_probabilities(
        X,
        sample_shape=torch.Size([8]),
    )[..., 1]
    torch.manual_seed(7)
    actual = sample_feasibility.probability(X, sample_shape=torch.Size([8]))

    assert (
        sample_feasibility.kind is FeasibilityRepresentationKind.SAMPLE_PROBABILITY_OF_FEASIBILITY
    )
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_sample_classification_feasibility_rejects_non_probability_binding() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(0, threshold=1.0),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]

    with pytest.raises(TypeError, match="ProbabilityOfFeasibility"):
        make_sample_classification_feasibility_bridge(model, binding)


def test_sample_classification_feasibility_rejects_deterministic_non_gp_classifier() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    classifier = RandomForestBinaryClassifier(
        train_x,
        torch.tensor([0, 1, 1]),
        n_estimators=4,
        random_state=0,
    )
    classifier.fit()
    model = HeterogeneousModel(regression, classifier)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ClassificationConstraint(1, feasible_class=1),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]

    with pytest.raises(TypeError, match="epistemic class-probability samples"):
        make_sample_classification_feasibility_bridge(model, binding)


@pytest.mark.parametrize(
    ("sample_shape", "batch_shape", "q", "expected"),
    [
        (torch.Size([8]), torch.Size(), 3, torch.Size([8, 3])),
        (torch.Size([4, 2]), torch.Size([5]), 3, torch.Size([4, 2, 5, 3])),
        (torch.Size([1]), torch.Size([2, 5]), 1, torch.Size([1, 2, 5, 1])),
    ],
)
def test_sample_shape_contract_preserves_resolved_sample_batch_and_q_dimensions(
    sample_shape: torch.Size,
    batch_shape: torch.Size,
    q: int,
    expected: torch.Size,
) -> None:
    contract = SampleShapeContract.from_resolved_shapes(
        sample_shape=sample_shape,
        batch_shape=batch_shape,
        q=q,
    )

    assert contract.sample_shape == sample_shape
    assert contract.batch_shape == batch_shape
    assert contract.q == q
    assert contract.value_shape == expected


def test_sample_shape_contract_rejects_empty_q() -> None:
    with pytest.raises(ValueError, match="q must be at least 1"):
        SampleShapeContract.from_resolved_shapes(
            sample_shape=torch.Size([8]),
            batch_shape=torch.Size(),
            q=0,
        )


def test_regression_objective_preserves_mc_batch_and_q_dimensions() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    model = HeterogeneousModel(SingleTaskGP(train_x, torch.sin(train_x)))
    semantics = ProblemSemantics(objectives=(RegressionObjective(0),))
    binding = resolve_acquisition_composition(model, semantics).objectives[0]
    objective = make_botorch_objective_bridge(model, binding).objective
    samples = torch.zeros(8, 2, 3, 1, dtype=torch.double)

    assert objective(samples).shape == torch.Size([8, 2, 3])


def test_continuous_constraint_preserves_mc_batch_and_q_dimensions() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    model = HeterogeneousModel(SingleTaskGP(train_x, torch.sin(train_x)))
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(0, threshold=1.0),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]
    constraint = make_continuous_constraint_bridge(model, binding)
    samples = torch.zeros(8, 2, 3, 1, dtype=torch.double)

    assert constraint(samples).shape == torch.Size([8, 2, 3])


def test_sample_classification_feasibility_preserves_batch_and_q_dimensions() -> None:
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
    feasibility = make_sample_classification_feasibility_bridge(model, binding)
    X = torch.tensor(
        [[[[0.1], [0.2], [0.3]]], [[[0.7], [0.8], [0.9]]]],
        dtype=torch.double,
    )
    X = X.squeeze(1)
    sample_shape = torch.Size([8])

    actual = feasibility.probability(X, sample_shape=sample_shape)
    expected = SampleShapeContract.from_resolved_shapes(
        sample_shape=sample_shape,
        batch_shape=torch.Size(X.shape[:-2]),
        q=X.shape[-2],
    ).value_shape

    assert actual.shape == expected


@pytest.mark.parametrize(
    ("direction", "best_f", "expected_best_f"),
    [
        (ObjectiveDirection.MAXIMIZE, 0.5, 0.5),
        (ObjectiveDirection.MINIMIZE, 0.5, -0.5),
    ],
)
def test_qei_integration_returns_native_botorch_acquisition(
    direction: ObjectiveDirection,
    best_f: float,
    expected_best_f: float,
) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0, direction=direction),),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))

    acquisition = make_qei_acquisition(
        model,
        semantics,
        best_f=best_f,
        sampler=sampler,
    )

    assert type(acquisition) is qExpectedImprovement
    assert acquisition.model is regression
    assert acquisition.sampler is sampler
    assert torch.equal(
        torch.as_tensor(acquisition.best_f),
        torch.tensor(expected_best_f, dtype=torch.double),
    )
    assert acquisition(torch.tensor([[0.25]], dtype=torch.double)).shape == torch.Size([1])


def test_qei_integration_rejects_constraint_composition() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(0, threshold=1.0),),
    )

    with pytest.raises(ValueError, match="unconstrained"):
        make_qei_acquisition(model, semantics, best_f=0.0)


def test_qei_integration_rejects_multiple_objectives() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    first = SingleTaskGP(train_x, torch.sin(train_x))
    second = SingleTaskGP(train_x, torch.cos(train_x))
    model = HeterogeneousModel(first, second)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0), RegressionObjective(1)),
    )

    with pytest.raises(ValueError, match="exactly one objective"):
        make_qei_acquisition(model, semantics, best_f=0.0)


def test_continuous_constrained_qei_uses_native_shared_posterior_objective() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_y = torch.cat((torch.sin(train_x), torch.cos(train_x)), dim=-1)
    regression = SingleTaskGP(train_x, train_y)
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(1, threshold=0.8),),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))

    acquisition = make_continuous_constrained_qei_acquisition(
        model,
        semantics,
        best_f=0.5,
        sampler=sampler,
    )

    assert type(acquisition) is qExpectedImprovement
    assert acquisition.model is regression
    assert acquisition.sampler is sampler
    assert isinstance(acquisition.objective, ConstrainedMCObjective)
    assert acquisition(torch.tensor([[0.25]], dtype=torch.double)).shape == torch.Size([1])


def test_continuous_constrained_qei_rejects_independent_model_constraint() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    objective_model = SingleTaskGP(train_x, torch.sin(train_x))
    constraint_model = SingleTaskGP(train_x, torch.cos(train_x))
    model = HeterogeneousModel(objective_model, constraint_model)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(1, threshold=0.8),),
    )

    with pytest.raises(ValueError, match="share one BoTorch posterior"):
        make_continuous_constrained_qei_acquisition(model, semantics, best_f=0.5)


def test_continuous_constrained_qei_rejects_classification_constraint() -> None:
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

    with pytest.raises(TypeError, match="continuous constraints only"):
        make_continuous_constrained_qei_acquisition(model, semantics, best_f=0.5)


def test_mixed_constrained_qei_combines_continuous_mc_and_classifier_pof() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_y = torch.cat((torch.sin(train_x), torch.cos(train_x)), dim=-1)
    regression = SingleTaskGP(train_x, train_y)
    classifier = BinarySingleTaskGPClassifier(
        train_x,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.double),
    )
    model = HeterogeneousModel(regression, classifier)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(
            ContinuousConstraint(1, threshold=0.8),
            ClassificationConstraint(2, feasible_class=1),
        ),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))

    acquisition = make_mixed_constrained_qei_acquisition(
        model,
        semantics,
        best_f=0.5,
        sampler=sampler,
    )

    assert isinstance(acquisition, FeasibilityWeightedAcquisition)
    assert type(acquisition.objective_acquisition) is qExpectedImprovement
    assert isinstance(acquisition.objective_acquisition.objective, ConstrainedMCObjective)
    assert acquisition.objective_acquisition.sampler is sampler
    assert acquisition(torch.tensor([[0.25]], dtype=torch.double)).shape == torch.Size([1])


def test_mixed_constrained_qei_rejects_independent_continuous_constraint() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    objective_model = SingleTaskGP(train_x, torch.sin(train_x))
    constraint_model = SingleTaskGP(train_x, torch.cos(train_x))
    classifier = BinarySingleTaskGPClassifier(
        train_x,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.double),
    )
    model = HeterogeneousModel(objective_model, constraint_model, classifier)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(
            ContinuousConstraint(1, threshold=0.8),
            ClassificationConstraint(2, feasible_class=1),
        ),
    )

    with pytest.raises(ValueError, match="share the objective"):
        make_mixed_constrained_qei_acquisition(model, semantics, best_f=0.5)


@pytest.mark.parametrize("constraint_kind", ["continuous", "classification"])
def test_mixed_constrained_qei_requires_both_constraint_kinds(constraint_kind: str) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.cat((torch.sin(train_x), torch.cos(train_x)), -1))
    classifier = BinarySingleTaskGPClassifier(
        train_x,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.double),
    )
    model = HeterogeneousModel(regression, classifier)
    constraint = (
        ContinuousConstraint(1, threshold=0.8)
        if constraint_kind == "continuous"
        else ClassificationConstraint(2, feasible_class=1)
    )
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(constraint,),
    )

    with pytest.raises(ValueError, match="both continuous and classification"):
        make_mixed_constrained_qei_acquisition(model, semantics, best_f=0.5)


@pytest.mark.parametrize("direction", [ObjectiveDirection.MAXIMIZE, ObjectiveDirection.MINIMIZE])
def test_qnei_integration_uses_native_botorch_acquisition(
    direction: ObjectiveDirection,
) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0, direction=direction),),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))

    acquisition = make_qnei_acquisition(
        model,
        semantics,
        X_baseline=train_x,
        sampler=sampler,
        cache_root=False,
    )

    assert type(acquisition) is qNoisyExpectedImprovement
    assert acquisition.model is regression
    assert acquisition.sampler is sampler
    assert acquisition(torch.tensor([[0.25]], dtype=torch.double)).shape == torch.Size([1])


def test_qnei_integration_rejects_constraints_and_empty_baseline() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    model = HeterogeneousModel(regression)
    constrained = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(0, threshold=1.0),),
    )
    with pytest.raises(ValueError, match="unconstrained"):
        make_qnei_acquisition(model, constrained, X_baseline=train_x)

    unconstrained = ProblemSemantics(objectives=(RegressionObjective(0),))
    with pytest.raises(ValueError, match="nonempty"):
        make_qnei_acquisition(model, unconstrained, X_baseline=train_x[:0])


def test_continuous_constrained_qnei_uses_native_mc_objective() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_y = torch.cat((torch.sin(train_x), torch.cos(train_x)), dim=-1)
    regression = SingleTaskGP(train_x, train_y)
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(1, threshold=0.8),),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))

    acquisition = make_continuous_constrained_qnei_acquisition(
        model,
        semantics,
        X_baseline=train_x,
        sampler=sampler,
        cache_root=False,
    )

    assert type(acquisition) is qNoisyExpectedImprovement
    assert acquisition.model is regression
    assert acquisition.sampler is sampler
    assert isinstance(acquisition.objective, ConstrainedMCObjective)
    assert acquisition(torch.tensor([[0.25]], dtype=torch.double)).shape == torch.Size([1])


def test_continuous_constrained_qnei_rejects_separate_constraint_model() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    objective_model = SingleTaskGP(train_x, torch.sin(train_x))
    constraint_model = SingleTaskGP(train_x, torch.cos(train_x))
    model = HeterogeneousModel(objective_model, constraint_model)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(1, threshold=0.8),),
    )
    with pytest.raises(ValueError, match="shared BoTorch posterior"):
        make_continuous_constrained_qnei_acquisition(
            model,
            semantics,
            X_baseline=train_x,
        )


def test_continuous_constrained_qnei_rejects_classification_constraint() -> None:
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
    with pytest.raises(TypeError, match="continuous constraints only"):
        make_continuous_constrained_qnei_acquisition(
            model,
            semantics,
            X_baseline=train_x,
        )


def test_multiple_learned_constrained_qnei_composes_two_classifiers() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    labels = torch.tensor([0.0, 1.0, 1.0], dtype=torch.double)
    model = HeterogeneousModel(
        regression,
        BinarySingleTaskGPClassifier(train_x, labels),
        BinarySingleTaskGPClassifier(train_x, labels),
    )
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(
            ClassificationConstraint(1, feasible_class=1),
            ClassificationConstraint(2, feasible_class=1),
        ),
    )

    acquisition = make_multiple_learned_constrained_qnei_acquisition(
        model,
        semantics,
        X_baseline=train_x,
        cache_root=False,
    )

    assert isinstance(acquisition, FeasibilityWeightedAcquisition)
    assert isinstance(acquisition.probability_of_feasibility, IndependentFeasibilityAggregator)
    assert type(acquisition.objective_acquisition) is qNoisyExpectedImprovement
    value = acquisition(torch.tensor([[0.25]], dtype=torch.double))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_multiple_learned_constrained_qnei_rejects_cross_entry_residual() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    model = HeterogeneousModel(
        SingleTaskGP(train_x, torch.sin(train_x)),
        SingleTaskGP(train_x, torch.cos(train_x)),
    )
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(ContinuousConstraint(1, threshold=0.8),),
    )
    with pytest.raises(ValueError, match="share"):
        make_multiple_learned_constrained_qnei_acquisition(
            model,
            semantics,
            X_baseline=train_x,
        )


def test_multiple_learned_constrained_qnei_rejects_duplicate_classifier_output() -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    regression = SingleTaskGP(train_x, torch.sin(train_x))
    classifier = BinarySingleTaskGPClassifier(
        train_x,
        torch.tensor([0.0, 1.0, 1.0], dtype=torch.double),
    )
    model = HeterogeneousModel(regression, classifier)
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=(
            ClassificationConstraint(1, feasible_class=1),
            ClassificationConstraint(1, feasible_class=0),
        ),
    )
    with pytest.raises(ValueError, match="Repeated classification"):
        make_multiple_learned_constrained_qnei_acquisition(
            model,
            semantics,
            X_baseline=train_x,
        )


@pytest.mark.parametrize("feasible_classes", [(1, 1), (0, 1)])
def test_multiple_learned_qnei_rejects_repeated_classifier_output(
    feasible_classes: tuple[int, int],
) -> None:
    train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    labels = torch.tensor([0.0, 1.0, 1.0], dtype=torch.double)
    model = HeterogeneousModel(
        SingleTaskGP(train_x, torch.sin(train_x)),
        BinarySingleTaskGPClassifier(train_x, labels),
    )
    semantics = ProblemSemantics(
        objectives=(RegressionObjective(0),),
        constraints=tuple(
            ClassificationConstraint(1, feasible_class=feasible_class)
            for feasible_class in feasible_classes
        ),
    )
    with pytest.raises(ValueError, match="Repeated classification output"):
        make_multiple_learned_constrained_qnei_acquisition(
            model,
            semantics,
            X_baseline=train_x,
        )


def test_independent_feasibility_aggregator_preserves_q_axis() -> None:
    from torch import nn

    class ConstantProbability(nn.Module):
        def __init__(self, probability: float) -> None:
            super().__init__()
            self.probability = probability

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            return X.new_full(X.shape[:-1], self.probability)

    aggregator = IndependentFeasibilityAggregator(
        [ConstantProbability(0.5), ConstantProbability(0.8)]
    )
    X = torch.zeros(2, 3, 1)
    assert torch.allclose(aggregator(X), torch.full((2, 3), 0.4))


def test_independent_feasibility_aggregator_rejects_empty_factors() -> None:
    with pytest.raises(ValueError, match="At least one"):
        IndependentFeasibilityAggregator([])
