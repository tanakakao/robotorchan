"""Study q-batch and mixed-variable candidate integration tests."""

import pytest
import torch

from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.study import TrialState, create_study


def test_ask_batch_initial_candidates_and_out_of_order_results() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=17)
    trials = study.ask_batch(3)
    assert len(trials) == 3
    assert [trial.number for trial in trials] == [0, 1, 2]
    assert all(trial.state is TrialState.RUNNING for trial in trials)
    for trial in reversed(trials):
        study.tell(trial, trial.params["x"])
    assert all(trial.state is TrialState.COMPLETE for trial in study.trials)


def test_ask_batch_rejects_invalid_q() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    with pytest.raises(ValueError, match="q must"):
        study.ask_batch(0)
    with pytest.raises(ValueError, match="q must"):
        study.ask_batch(True)


def test_mixed_initial_batch_respects_discrete_domains() -> None:
    bounds = torch.tensor([[0.0, 1.0, 0.0], [1.0, 5.0, 2.0]], dtype=torch.double)
    variable_space = MixedVariableSpace(
        bounds=bounds,
        integer_dims=(1,),
        categorical_values={2: (0.0, 2.0)},
    )
    study = create_study(
        search_space={"x": (0.0, 1.0), "count": (1.0, 5.0), "category": (0.0, 2.0)},
        variable_space=variable_space,
        seed=8,
    )
    trials = study.ask_batch(5)
    assert len(trials) == 5
    for trial in trials:
        assert trial.params["count"].is_integer()
        assert trial.params["category"] in (0.0, 2.0)


def test_mixed_bounds_mismatch_rejected() -> None:
    variable_space = MixedVariableSpace(
        bounds=torch.tensor([[0.0], [2.0]], dtype=torch.double),
        integer_dims=(0,),
    )
    with pytest.raises(ValueError, match="bounds must match"):
        create_study(search_space={"x": (0.0, 1.0)}, variable_space=variable_space)
