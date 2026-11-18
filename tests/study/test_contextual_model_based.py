"""Model-based contextual Study dispatch contracts."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.study import create_study


class _FittedModel:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y

    def fit(self):
        return self


class _Acquisition:
    def __init__(self, model, best_f):
        self.model = model
        self.best_f = best_f


def test_model_based_mixed_q_batch_passes_context_to_dispatch() -> None:
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 3.0, 2.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds=bounds, integer_dims=(1,), categorical_values={2: (0.0, 2.0)})
    configs = [{1: 0.0, 2: 0.0}, {1: 1.0, 2: 2.0}]
    study = create_study(
        search_space={"decision": (0.0, 1.0), "integer": (0.0, 3.0), "category": (0.0, 2.0)},
        context_names=("integer",),
        variable_space=space,
        model=_FittedModel,
        acquisition=_Acquisition,
        optimizer_kwargs={"optimizer_options": {"fixed_features_list": configs}},
        n_initial_trials=1,
        seed=7,
    )
    first = study.ask(context={"integer": 1.0})
    study.tell(first, 0.3)
    expected = torch.tensor([[0.4, 1.0, 0.0], [0.7, 1.0, 2.0]], dtype=torch.double)
    with patch("robotorchan.study.generation.optimize_acqf") as optimize:
        optimize.return_value = (expected, torch.tensor(0.0))
        trials = study.ask_batch(2, context={"integer": 1.0})
    assert len(trials) == 2
    assert all(trial.params["integer"] == 1.0 for trial in trials)
    assert optimize.call_args.kwargs["fixed_features"] == {1: 1.0}
    assert optimize.call_args.kwargs["variable_space"] is space
    assert optimize.call_args.kwargs["optimizer_options"]["fixed_features_list"] == configs


def test_model_based_context_rejects_optimizer_context_override() -> None:
    study = create_study(
        search_space={"decision": (0.0, 1.0), "context": (0.0, 1.0)},
        context_names=("context",),
        model=_FittedModel,
        acquisition=_Acquisition,
        optimizer_kwargs={"fixed_features": {1: 0.1}},
        n_initial_trials=1,
        seed=8,
    )
    first = study.ask(context={"context": 0.5})
    study.tell(first, 0.1)
    with pytest.raises(ValueError, match="override context"):
        study.ask(context={"context": 0.5})


def test_model_based_context_rejects_backend_context_violation() -> None:
    study = create_study(
        search_space={"decision": (0.0, 1.0), "context": (0.0, 1.0)},
        context_names=("context",),
        model=_FittedModel,
        acquisition=_Acquisition,
        n_initial_trials=1,
        seed=8,
    )
    first = study.ask(context={"context": 0.5})
    study.tell(first, 0.1)
    with patch("robotorchan.study.generation.optimize_acqf") as optimize:
        optimize.return_value = (
            torch.tensor([[0.4, 0.6]], dtype=torch.double),
            torch.tensor(0.0),
        )
        with pytest.raises(ValueError, match="preserve"):
            study.ask(context={"context": 0.5})
