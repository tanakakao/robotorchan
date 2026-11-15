"""Feasible Study initialization and external trial validation."""

import pytest
import torch

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.study import create_study


def test_initial_linear_inequality_and_external_trial_validation() -> None:
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.8),),
    )
    study = create_study(
        search_space={"x": (0.0, 1.0)},
        constraints=constraints,
        seed=7,
    )
    with pytest.raises(ValueError, match="CandidateConstraints"):
        study.add_trial({"x": 0.2})
    trials = study.ask_batch(3)
    assert len(trials) == 3
    assert all(trial.params["x"] >= 0.8 - 1e-7 for trial in trials)


def test_interpoint_batch_feasible() -> None:
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([1.0, 1.0]),
                1.5,
            ),
        ),
    )
    study = create_study(
        search_space={"x": (0.0, 1.0)},
        constraints=constraints,
        seed=3,
    )
    trials = study.ask_batch(2)
    assert sum(trial.params["x"] for trial in trials) >= 1.5 - 1e-7


def test_mixed_feasible_initial_design() -> None:
    variable_space = MixedVariableSpace(
        bounds=torch.tensor([[0.0, 1.0], [1.0, 4.0]], dtype=torch.double),
        integer_dims=(1,),
    )
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([1]), torch.tensor([1.0]), 3.0),),
    )
    study = create_study(
        search_space={"x": (0.0, 1.0), "count": (1.0, 4.0)},
        variable_space=variable_space,
        constraints=constraints,
        seed=1,
    )
    with pytest.raises(ValueError, match="integral"):
        study.add_trial({"x": 0.5, "count": 2.5})
    assert all(trial.params["count"] >= 3 for trial in study.ask_batch(3))


def test_infeasible_initial_space_fails_bounded() -> None:
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 2.0),),
    )
    study = create_study(search_space={"x": (0.0, 1.0)}, constraints=constraints)
    with pytest.raises(RuntimeError, match="256 attempts"):
        study.ask()
    assert not study.trials
