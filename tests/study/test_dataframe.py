"""DataFrame export for Study history."""

import pytest

from robotorchan.study import TrialState, create_study

pd = pytest.importorskip("pandas")


def test_dataframe_columns_and_states() -> None:
    study = create_study(search_space={"temperature": (100.0, 300.0), "pressure": (1.0, 10.0)})
    first = study.add_trial({"temperature": 150.0, "pressure": 2.0})
    second = study.add_trial({"temperature": 200.0, "pressure": 4.0})
    study.start_trial(first.number)
    study.tell(first.number, 0.75)
    study.start_trial(second.number)
    study.tell(second.number, state=TrialState.FAIL)

    frame = study.trials_dataframe()
    assert list(frame.columns) == ["number", "state", "value", "temperature", "pressure"]
    assert frame["number"].tolist() == [0, 1]
    assert frame["state"].tolist() == ["COMPLETE", "FAIL"]
    assert frame.loc[0, "value"] == 0.75
    assert pd.isna(frame.loc[1, "value"])
    assert frame["temperature"].tolist() == [150.0, 200.0]


def test_dataframe_empty_and_filtered() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    empty = study.trials_dataframe()
    assert empty.empty
    assert list(empty.columns) == ["number", "state", "value", "x"]
    trial = study.ask()
    study.tell(trial, 0.25)
    assert study.trials_dataframe(states=(TrialState.FAIL,)).empty
    assert study.trials_dataframe(states=(TrialState.COMPLETE,))["value"].tolist() == [0.25]


def test_dataframe_rejects_invalid_state_filter() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    with pytest.raises(TypeError, match="states"):
        study.trials_dataframe(states=("COMPLETE",))


def test_dataframe_is_snapshot_not_mutable_study() -> None:
    study = create_study(search_space={"x": (0.0, 1.0)})
    trial = study.ask()
    study.tell(trial, 0.5)
    frame = study.trials_dataframe()
    frame.loc[0, "value"] = 100.0
    assert study.best_value == 0.5
