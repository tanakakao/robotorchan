"""Different-context candidate proposals use independent q=1 optimization."""

from unittest.mock import patch

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


class _Model:
    def __init__(self, train_x, train_y):
        self.train_x = train_x

    def fit(self):
        return self


class _Acquisition:
    def __init__(self, model, best_f):
        self.model = model


def test_ask_contexts_model_based_dispatches_independent_q1() -> None:
    study = create_study(
        search_space={"decision": (0.0, 1.0), "ambient": (0.0, 1.0)},
        context_names=("ambient",),
        model=_Model,
        acquisition=_Acquisition,
        n_initial_trials=1,
        seed=10,
    )
    first = study.ask(context={"ambient": 0.3})
    study.tell(first, 0.5)
    with patch("robotorchan.optim.factory.optimize_acqf") as optimize:
        optimize.side_effect = [
            (torch.tensor([[0.4, 0.2]], dtype=torch.double), torch.tensor(0.0)),
            (torch.tensor([[0.6, 0.8]], dtype=torch.double), torch.tensor(0.0)),
        ]
        trials = study.ask_contexts([{"ambient": 0.2}, {"ambient": 0.8}])
    assert len(trials) == 2
    assert [call.kwargs["q"] for call in optimize.call_args_list] == [1, 1]
    assert [call.kwargs["fixed_features"] for call in optimize.call_args_list] == [
        {1: 0.2},
        {1: 0.8},
    ]
