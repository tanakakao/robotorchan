"""Regression tests for decoupled workflow requirements."""

import pytest

from robotorchan.workflow import ProblemSpec, check_compatibility


def test_decoupled_requires_multi_output():
    with pytest.raises(ValueError, match="multi-output"):
        ProblemSpec(purpose="bayesian_optimization", decoupled=True)


def test_pending_requires_decoupled():
    with pytest.raises(ValueError, match="requires decoupled"):
        ProblemSpec(purpose="active_learning", masked_pending=True)


def test_pending_rejects_unsupported_acquisition():
    spec = ProblemSpec(
        purpose="bayesian_optimization",
        output_type="multi",
        decoupled=True,
        masked_pending=True,
    )
    decision = check_compatibility(
        spec,
        model="HeterogeneousModel",
        acquisition="qDecoupledParticleKnowledgeGradient",
        optimizer="optimize_acqf_decoupled",
    )
    assert "requires masked pending evaluation support" in decision.reasons


def test_decoupled_rejects_scalar_optimizer():
    spec = ProblemSpec(
        purpose="bayesian_optimization",
        output_type="multi",
        decoupled=True,
    )
    decision = check_compatibility(
        spec,
        model="HeterogeneousModel",
        acquisition="qDecoupledParticleKnowledgeGradient",
        optimizer="botorch",
    )
    assert "requires the dedicated decoupled optimizer without a search strategy" in (
        decision.reasons
    )


@pytest.mark.parametrize(
    ("name", "pending"),
    [
        ("qHypervolumeKnowledgeGradient", True),
        ("qMultiFidelityHypervolumeKnowledgeGradient", True),
        ("qDecoupledKnowledgeGradient", False),
        ("qDecoupledExpectedPredictiveInformationGain", True),
        ("qDecoupledActiveLearning", False),
        ("qDecoupledParticleKnowledgeGradient", False),
    ],
)
def test_masked_registry_capabilities(name, pending):
    from robotorchan.acquisition.registry import ACQUISITION_REGISTRY

    caps = ACQUISITION_REGISTRY[name].capabilities
    assert caps.supports_decoupled
    assert caps.supports_masked_pending is pending


def test_recommendation_driven_proposer():
    import torch

    from robotorchan.study.decoupled import create_decoupled_proposer
    from robotorchan.workflow import recommend

    spec = ProblemSpec(
        purpose="bayesian_optimization",
        output_type="multi",
        decoupled=True,
    )
    recommendations = recommend(
        spec,
        model="HeterogeneousModel",
        acquisition="qDecoupledParticleKnowledgeGradient",
    )
    assert recommendations
    recommendation = recommendations[0]
    assert recommendation.optimizer_name == "optimize_acqf_decoupled"
    kwargs = dict(
        parameter_names=("x",),
        output_names=("a", "b"),
        evaluation_masks=torch.eye(2, dtype=torch.bool),
        model_factory=lambda data: None,
        recommendation=recommendation,
        problem_spec=spec,
    )
    proposer = create_decoupled_proposer(**kwargs)
    with pytest.raises(ValueError, match="Requested q"):
        proposer([], torch.tensor([[0.0], [1.0]]), q=2, fixed_features={})
    with pytest.raises(ValueError, match="cannot accompany"):
        create_decoupled_proposer(
            **kwargs, acquisition_factory="qDecoupledParticleKnowledgeGradient"
        )
    with pytest.raises(ValueError, match="masked_pending"):
        create_decoupled_proposer(**kwargs, pending_fantasies=True)


@pytest.mark.parametrize("invalid", [(0.5, 0.0), (1.0, 1.0)])
def test_invalid_mixed_batch_leaves_no_pending_trials(invalid):
    import torch

    from robotorchan.optim.domains.variable_space import MixedVariableSpace
    from robotorchan.study import create_study

    bounds = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, integer_dims=(0,), categorical_values={1: (0.0, 2.0)})
    rows = [(0.0, 0.0), invalid]

    def propose(history, bounds, *, q, fixed_features):
        return (
            bounds.new_tensor(rows),
            torch.ones((q, 2), dtype=torch.bool),
            bounds.new_tensor(0.0),
        )

    study = create_study(
        search_space={"integer": (0.0, 2.0), "category": (0.0, 2.0)},
        variable_space=space,
        output_names=("a", "b"),
        decoupled_proposer=propose,
    )
    with pytest.raises(ValueError, match=r"integer|categorical"):
        study.ask_batch(2)
    assert not study.trials
    rows[1] = (2.0, 2.0)
    trials = study.ask_batch(2)
    assert [trial.number for trial in trials] == [0, 1]


def test_recommended_gp_decoupled_ask_tell():
    import torch
    from botorch.models import ModelListGP, SingleTaskGP

    from robotorchan.study import create_study
    from robotorchan.study.decoupled import create_decoupled_proposer
    from robotorchan.workflow import recommend

    spec = ProblemSpec(
        purpose="active_learning",
        output_type="multi",
        decoupled=True,
    )
    recommendations = recommend(
        spec,
        model="ModelListGP",
        acquisition="qDecoupledExpectedPredictiveInformationGain",
    )
    assert recommendations
    X = torch.linspace(0, 1, 5, dtype=torch.double).unsqueeze(-1)
    masks = torch.eye(2, dtype=torch.bool)

    def model_factory(data):
        models = []
        for name in ("a", "b"):
            train_X, train_Y = data[name]
            assert len(train_X) >= 5
            models.append(SingleTaskGP(train_X, train_Y))
        model = ModelListGP(*models)
        model.eval()
        return model

    propose = create_decoupled_proposer(
        parameter_names=("x",),
        output_names=("a", "b"),
        evaluation_masks=masks,
        model_factory=model_factory,
        recommendation=recommendations[0],
        problem_spec=spec,
        acquisition_kwargs={
            "target_X": X,
            "X_evaluation_mask": masks[:1],
            "observation_noise": X.new_tensor([0.1, 0.1]),
        },
        optimizer_kwargs={"optimizer": "sobol", "optimizer_options": {"num_samples": 8}},
    )
    study = create_study(
        search_space={"x": (0.0, 1.0)},
        output_names=("a", "b"),
        decoupled_proposer=propose,
    )
    for x in X.flatten():
        trial = study.add_trial({"x": x.item()}, evaluation_mask=("a", "b"))
        study.start_trial(trial.number)
        study.tell(trial, observations={"a": x.sin().item(), "b": x.cos().item()})
    for _ in range(2):
        trial = study.ask()
        assert len(trial.evaluation_mask) == 1
        assert 0 <= trial.params["x"] <= 1
        study.tell(trial, observations={trial.evaluation_mask[0]: 0.5})
    assert len(study.trials) == 7
