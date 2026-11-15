"""Optuna-like Study loop and best-result behavior."""

import pytest

from robotorchan.study import TrialState, create_study


def test_optimize_callbacks_and_best_maximize() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=2)
    observed = []

    def callback(current_study, trial):
        assert current_study is study
        observed.append((trial.number, trial.state))

    study.optimize(lambda trial: trial.params["x"], n_trials=3, callbacks=[callback])
    assert len(study.trials) == 3
    assert observed == [(i, TrialState.COMPLETE) for i in range(3)]
    assert study.best_value == max(trial.value for trial in study.trials)
    assert study.best_params == study.best_trial.params


def test_best_minimize_and_empty_study() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, direction="minimize")
    with pytest.raises(ValueError, match="No completed trials"):
        _ = study.best_trial
    study.optimize(lambda trial: trial.params["x"], n_trials=2)
    assert study.best_value == min(trial.value for trial in study.trials)


def test_caught_failure_continues_and_callbacks_receive_failed_trial() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=4)
    states = []

    def objective(trial):
        if trial.number == 0:
            raise RuntimeError("experiment failed")
        return trial.params["x"]

    study.optimize(
        objective,
        n_trials=3,
        catch=(RuntimeError,),
        callbacks=[lambda _study, trial: states.append(trial.state)],
    )
    assert states == [TrialState.FAIL, TrialState.COMPLETE, TrialState.COMPLETE]
    assert study.best_trial.number in {1, 2}


def test_uncaught_exception_marks_trial_failed_and_propagates() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    with pytest.raises(RuntimeError, match="failure"):
        study.optimize(lambda _trial: (_ for _ in ()).throw(RuntimeError("failure")), 1)
    assert study.trials[0].state is TrialState.FAIL


def test_optimize_validates_arguments_and_zero_trials() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    study.optimize(lambda trial: trial.params["x"], 0)
    assert not study.trials
    with pytest.raises(ValueError, match="n_trials"):
        study.optimize(lambda trial: trial.params["x"], -1)
    with pytest.raises(TypeError, match="catch"):
        study.optimize(lambda trial: trial.params["x"], 1, catch=(1,))
    with pytest.raises(TypeError, match="callbacks"):
        study.optimize(lambda trial: trial.params["x"], 1, callbacks=[None])
