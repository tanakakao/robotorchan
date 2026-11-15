"""Focused tests for the Phase 2 in-memory Study core."""

import pytest

from robotorchan.study import SearchSpace, TrialState, create_study


def test_trial_lifecycle_and_stable_ids() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    first = study.add_trial({"x": 0.25})
    second = study.add_trial({"x": 0.75})
    assert (first.number, second.number) == (0, 1)
    assert first.state is TrialState.WAITING
    running = study.start_trial(first.number)
    assert running.state is TrialState.RUNNING
    complete = study.complete_trial(first.number, 2.5)
    assert complete.state is TrialState.COMPLETE
    assert complete.value == 2.5
    assert first.state is TrialState.WAITING
    assert [trial.number for trial in study.trials] == [0, 1]
    assert study.get_trial(0) == complete


def test_failure_and_invalid_transitions() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    trial = study.add_trial({"x": 0.5})
    with pytest.raises(ValueError, match="must be RUNNING"):
        study.complete_trial(trial.number, 1.0)
    study.start_trial(trial.number)
    study.fail_trial(trial.number)
    with pytest.raises(ValueError, match="must be RUNNING"):
        study.complete_trial(trial.number, 1.0)
    with pytest.raises(ValueError, match="must be WAITING"):
        study.start_trial(trial.number)


def test_reject_invalid_candidates_and_observations() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    for params in ({"x": 1.5}, {}, {"x": float("nan")}, {"x": True}):
        with pytest.raises(ValueError):
            study.add_trial(params)
    assert study.trials == ()
    trial = study.add_trial({"x": 0.5})
    study.start_trial(trial.number)
    with pytest.raises(ValueError, match="finite"):
        study.complete_trial(trial.number, float("nan"))
    study.complete_trial(trial.number, 1.0)
    with pytest.raises(ValueError, match="must be RUNNING"):
        study.complete_trial(trial.number, 2.0)


def test_search_space_validation_and_snapshots() -> None:
    for bounds in ({}, {"x": (1.0, 1.0)}, {"x": (0.0, float("inf"))}):
        with pytest.raises(ValueError):
            SearchSpace(bounds)
    params = {"x": 0.2}
    study = create_study(search_space={"x": (0.0, 1.0)})
    trial = study.add_trial(params)
    params["x"] = 0.8
    assert trial.params["x"] == 0.2
    with pytest.raises(TypeError):
        trial.params["x"] = 0.4
    with pytest.raises(KeyError):
        study.get_trial(999)
