"""Contextual Study integration with mixed variables, q-batches and constraints."""

import pytest
import torch

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.study import create_study


def test_mixed_context_and_decision_q_batch() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 4.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        bounds=bounds, integer_dims=(1,), categorical_values={2: (0.0, 2.0)}
    )
    study = create_study(
        search_space={"decision": (0.0, 1.0), "integer": (0.0, 4.0), "category": (0.0, 2.0)},
        context_names=("integer", "category"),
        variable_space=variable_space,
        optimizer="sobol",
        n_initial_trials=4,
        seed=10,
    )
    trials = study.ask_batch(3, context={"integer": 2.0, "category": 2.0})
    assert len(trials) == 3
    for trial in trials:
        assert trial.params["integer"] == 2.0
        assert trial.params["category"] == 2.0
        assert 0.0 <= trial.params["decision"] <= 1.0


def test_mixed_context_rejects_illegal_integer_and_category() -> None:
    bounds = torch.tensor([[0.0, 0.0], [4.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        bounds=bounds, integer_dims=(0,), categorical_values={1: (0.0, 2.0)}
    )
    study = create_study(
        search_space={"integer": (0.0, 4.0), "category": (0.0, 2.0)},
        context_names=("integer",),
        variable_space=variable_space,
    )
    with pytest.raises(ValueError, match="integer"):
        study.ask(context={"integer": 1.5})
    with pytest.raises(ValueError, match="finite"):
        study.ask(context={"integer": True})


def test_contextual_q_batch_respects_candidate_constraints() -> None:
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([0, 1]),
                torch.tensor([-1.0, 1.0], dtype=torch.double),
                0.2,
            ),
        ),
    )
    study = create_study(
        search_space={"decision": (0.0, 1.0), "context": (0.0, 1.0)},
        context_names=("context",),
        constraints=constraints,
        optimizer="sobol",
        n_initial_trials=4,
        seed=5,
    )
    trials = study.ask_batch(2, context={"context": 0.8})
    assert len(trials) == 2
    for trial in trials:
        assert trial.params["context"] == 0.8
        assert trial.params["decision"] <= 0.6 + 1e-10
