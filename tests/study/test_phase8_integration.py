"""Phase 8 end-to-end contracts for the public Study API."""

import pytest
import torch

from robotorchan.optim.constraints.contracts import CandidateConstraints
from robotorchan.optim.domains.variable_space import MixedVariableSpace
from robotorchan.study import TrialState, create_study


def test_ask_tell_optimize_and_best_trial_lifecycle() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=17, n_initial_trials=10)
    first = study.ask()
    assert first.state is TrialState.RUNNING
    assert study.tell(first, first.params["x"]).state is TrialState.COMPLETE
    study.optimize(lambda trial: trial.params["x"], n_trials=2)
    assert len(study.trials) == 3
    assert all(trial.state is TrialState.COMPLETE for trial in study.trials)
    assert study.best_value == max(trial.value for trial in study.trials)
    assert study.best_params == study.best_trial.params
    with pytest.raises(ValueError, match="must be RUNNING"):
        study.tell(first, 0.0)


def test_mixed_constrained_joint_batch_external_validation() -> None:
    space = MixedVariableSpace(
        bounds=torch.tensor([[0.0, 1.0], [1.0, 4.0]], dtype=torch.double),
        integer_dims=(1,),
    )
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([[0, 1], [1, 1]]), torch.tensor([1.0, 1.0]), 6.0),),
    )
    study = create_study(
        search_space={"x": (0.0, 1.0), "count": (1.0, 4.0)},
        variable_space=space,
        constraints=constraints,
        seed=11,
        n_initial_trials=10,
    )
    trials = study.ask_batch(2)
    assert len(trials) == 2
    assert sum(trial.params["count"] for trial in trials) >= 6
    assert all(trial.params["count"].is_integer() for trial in trials)
    for trial in trials:
        study.tell(trial.number, trial.params["x"])
    assert len(study.trials) == 2
    assert all(trial.state is TrialState.COMPLETE for trial in study.trials)


def test_external_trial_validation_preserves_trial_numbering() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=5)
    with pytest.raises(ValueError, match="outside"):
        study.add_trial({"x": 2.0})
    assert study.trials == ()
    trial = study.add_trial({"x": 0.5})
    assert trial.number == 0
    assert trial.state is TrialState.WAITING
    running = study.start_trial(trial.number)
    assert running.state is TrialState.RUNNING
    assert study.tell(running, state=TrialState.FAIL).state is TrialState.FAIL
    with pytest.raises(ValueError, match="No completed"):
        _ = study.best_trial


def test_dataframe_roundtrip_with_filtered_states() -> None:
    pd = pytest.importorskip("pandas")
    study = create_study(search_space={"x": (0.0, 1.0)}, seed=2)
    first = study.ask()
    study.tell(first, 0.75)
    second = study.ask()
    study.tell(second, state=TrialState.FAIL)
    frame = study.trials_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["number", "state", "value", "x"]
    assert list(frame["state"]) == ["COMPLETE", "FAIL"]
    assert frame.loc[0, "value"] == 0.75
    assert pd.isna(frame.loc[1, "value"])
    filtered = study.trials_dataframe(states=(TrialState.COMPLETE,))
    assert list(filtered["number"]) == [0]


def test_factory_study_resume_from_history() -> None:
    kwargs = {
        "search_space": {"x": (0.0, 1.0)},
        "model": "SingleTaskGP",
        "acquisition": "qLogExpectedImprovement",
        "optimizer": "sobol",
        "optimizer_kwargs": {"raw_samples": 16, "seed": 17},
        "n_initial_trials": 3,
        "seed": 17,
    }
    original = create_study(**kwargs)
    for _ in range(3):
        trial = original.ask()
        original.tell(trial, trial.params["x"])
    failed = original.ask()
    original.tell(failed, state=TrialState.FAIL)
    pending = original.ask()
    original.add_trial({"x": 0.5})

    resumed = create_study(**kwargs, initial_trials=original.trials)
    assert resumed.trials == original.trials
    assert resumed.best_value == original.best_value
    resumed.tell(pending.number, pending.params["x"])
    trials = resumed.ask_batch(2)
    assert [trial.number for trial in trials] == [6, 7]
    for trial in trials:
        resumed.tell(trial, trial.params["x"])
    assert len(resumed.trials) == 8


def test_resume_rejects_duplicate_history_and_invalid_domain() -> None:
    original = create_study(search_space={"x": (0.0, 1.0)})
    trial = original.add_trial({"x": 0.8})
    with pytest.raises(ValueError, match="strictly increasing"):
        create_study(search_space={"x": (0.0, 1.0)}, initial_trials=[trial, trial])
    with pytest.raises(ValueError, match="outside"):
        create_study(search_space={"x": (0.0, 0.5)}, initial_trials=[trial])
