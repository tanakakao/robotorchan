"""Different-context candidate proposals use independent q=1 optimization."""

import pytest
import torch

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.study import create_study


def test_ask_contexts_returns_one_trial_per_observed_context() -> None:
    study = create_study(
        search_space={"decision": (0.0, 1.0), "ambient": (0.0, 1.0)},
        context_names=("ambient",),
        n_initial_trials=4,
        seed=15,
    )
    contexts = [{"ambient": 0.2}, {"ambient": 0.8}, {"ambient": 0.4}]
    trials = study.ask_contexts(contexts)
    assert len(trials) == 3
    assert [trial.params["ambient"] for trial in trials] == pytest.approx([0.2, 0.8, 0.4])
    assert len({trial.number for trial in trials}) == 3


def test_ask_contexts_rejects_invalid_inputs_without_registering_trials() -> None:
    study = create_study(
        search_space={"decision": (0.0, 1.0), "ambient": (0.0, 1.0)},
        context_names=("ambient",),
    )
    with pytest.raises(ValueError, match="nonempty"):
        study.ask_contexts([])
    with pytest.raises(ValueError, match="Each context"):
        study.ask_contexts([{"ambient": 0.2}, {"decision": 0.5}])
    with pytest.raises(ValueError, match="outside"):
        study.ask_contexts([{"ambient": 0.2}, {"ambient": 2.0}])
    assert not study.trials


def test_ask_contexts_rejects_interpoint_constraints() -> None:
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([1.0, -1.0], dtype=torch.double),
                0.0,
            ),
        )
    )
    study = create_study(
        search_space={"decision": (0.0, 1.0), "ambient": (0.0, 1.0)},
        context_names=("ambient",),
        constraints=constraints,
    )
    with pytest.raises(NotImplementedError, match="inter-point"):
        study.ask_contexts([{"ambient": 0.2}, {"ambient": 0.8}])
    assert not study.trials
