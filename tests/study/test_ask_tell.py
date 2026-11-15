"""Focused Study ask/tell and configurable candidate-generation tests."""

import pytest

from robotorchan.study import TrialState, create_study


def test_sobol_ask_tell_and_reproducibility() -> None:
    first = create_study(search_space={"x": (0.0, 1.0)}, seed=7)
    second = create_study(search_space={"x": (0.0, 1.0)}, seed=7)
    a = first.ask()
    b = second.ask()
    assert a.params == b.params
    assert a.state is TrialState.RUNNING
    assert 0 <= a.params["x"] <= 1
    completed = first.tell(a, 0.5)
    assert completed.state is TrialState.COMPLETE
    assert completed.value == 0.5
    with pytest.raises(ValueError, match="RUNNING"):
        first.tell(a, 0.7)


def test_study_rejects_unsupported_settings() -> None:
    with pytest.raises(ValueError, match="direction"):
        create_study(search_space={"x": (0.0, 1.0)}, direction="invalid")
    with pytest.raises(ValueError, match="n_initial_trials"):
        create_study(search_space={"x": (0.0, 1.0)}, n_initial_trials=0)


def test_default_gp_acquisition_optimizer_loop() -> None:
    study = create_study(
        search_space={"x": (0.0, 1.0)},
        n_initial_trials=3,
        seed=3,
        optimizer_kwargs={"num_restarts": 2, "raw_samples": 16},
    )
    for _ in range(4):
        trial = study.ask()
        study.tell(trial, -((trial.params["x"] - 0.3) ** 2))
    assert len(study.trials) == 4
    assert all(trial.state is TrialState.COMPLETE for trial in study.trials)
