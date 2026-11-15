"""Study candidate-constraint integration and initialization safeguards."""

import pytest
import torch

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.study import create_study


def test_constrained_initial_ask_rejected_without_silent_violation() -> None:
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([1.0]), 0.8),),
    )
    study = create_study(
        search_space={"x": (0.0, 1.0)},
        constraints=constraints,
        n_initial_trials=1,
    )
    with pytest.raises(NotImplementedError, match="Constrained initial"):
        study.ask()
    assert not study.trials


def test_unconstrained_study_still_asks() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=9)
    assert study.ask().number == 0
