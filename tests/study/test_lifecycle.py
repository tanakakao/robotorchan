"""Focused lifecycle tests for external and concurrent Study evaluations."""

import pytest

from robotorchan.study import Trial, TrialState, create_study


def test_multiple_pending_trials_and_out_of_order_tell() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=11)
    first = study.ask()
    second = study.ask()
    assert first.number != second.number
    assert first.params != second.params
    assert all(t.state is TrialState.RUNNING for t in study.trials)

    study.tell(second.number, 0.2)
    study.tell(first, 0.8)
    assert [t.value for t in study.trials] == [0.8, 0.2]


def test_failed_trial_is_terminal_and_has_no_observation() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=5)
    failed = study.ask()
    result = study.tell(failed, state=TrialState.FAIL)
    assert result.state is TrialState.FAIL
    assert result.value is None
    with pytest.raises(ValueError, match="RUNNING"):
        study.tell(failed, 1.0)
    assert study.ask().state is TrialState.RUNNING


def test_tell_validates_terminal_state_and_value() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    trial = study.ask()
    with pytest.raises(ValueError, match="require an observation"):
        study.tell(trial)
    with pytest.raises(ValueError, match="must not contain"):
        study.tell(trial, 1.0, state=TrialState.FAIL)
    with pytest.raises(ValueError, match="only accepts"):
        study.tell(trial, state=TrialState.RUNNING)
    with pytest.raises(ValueError, match="parameters do not match"):
        study.tell(Trial(number=trial.number, params={"x": 0.0}), 0.2)
    assert study.get_trial(trial.number).state is TrialState.RUNNING


def test_external_candidate_can_be_started_and_reported() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    waiting = study.add_trial({"x": 0.5})
    assert waiting.state is TrialState.WAITING
    running = study.start_trial(waiting.number)
    result = study.tell(running.number, 0.75)
    assert result.state is TrialState.COMPLETE
    with pytest.raises(ValueError, match="RUNNING"):
        study.tell(running.number, 0.9)
