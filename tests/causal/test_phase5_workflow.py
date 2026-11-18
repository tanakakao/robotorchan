"""Focused Causal BO ask/tell tests using the existing model and trial states."""

import pytest

from robotorchan.causal import CausalBOStudy, CausalGraph, CausalObservation
from robotorchan.study import TrialState


def _workflow(*, sets=(("X",), ("Z",))):
    graph = CausalGraph({"X": (), "Z": (), "Y": ("X", "Z")})
    return CausalBOStudy(
        graph,
        "Y",
        sets,
        {"X": (0.0, 1.0), "Z": (0.0, 1.0)} if len(sets) == 2 else {"X": (0.0, 1.0)},
        n_initial_trials=2,
        optimizer="sobol",
        raw_samples=16,
        seed=12,
    )


def test_bootstrap_and_history_reject_duplicate_and_mismatched_tell():
    workflow = _workflow()
    expected = ("X", "Z", "X", "Z")
    for name in expected:
        trial = workflow.ask()
        assert trial.state is TrialState.RUNNING
        assert set(trial.intervention.values) == {name}
        with pytest.raises(RuntimeError, match="running"):
            workflow.ask()
        result = workflow.tell(trial, 0.5)
        assert result.state is TrialState.COMPLETE
        assert isinstance(result.observation, CausalObservation)
        assert result.observation.values["Y"] == 0.5
        with pytest.raises(ValueError, match="RUNNING"):
            workflow.tell(trial, 0.5)
    assert len(workflow.trials) == 4
    assert len(workflow.observations) == 4
    assert [trial.number for trial in workflow.trials] == [0, 1, 2, 3]


def test_failed_trial_retries_and_preserves_experiment_history():
    workflow = _workflow(sets=(("X",),))
    first = workflow.ask()
    failed = workflow.tell(first, state=TrialState.FAIL)
    assert failed.state is TrialState.FAIL
    assert not workflow.observations
    retry = workflow.ask()
    assert retry.number == 1
    assert set(retry.intervention.values) == {"X"}
    with pytest.raises(ValueError, match="match"):
        workflow.tell(retry, {"X": 9.0, "Y": 1.0})
    assert workflow.get_trial(retry.number).state is TrialState.RUNNING
    assert workflow.tell(retry, {"Y": 1.0}).state is TrialState.COMPLETE
    assert len(workflow.observations) == 1


def test_gp_selection_and_refit_after_completed_tell():
    workflow = _workflow(sets=(("X",),))
    for _ in range(2):
        trial = workflow.ask()
        x = trial.intervention.values["X"]
        workflow.tell(trial, x * x)

    models = workflow.fit()
    assert len(models) == 1
    assert workflow.fit()[0] is models[0]
    selected = workflow.ask()
    assert selected.proposal is not None
    assert 0.0 <= selected.intervention.values["X"] <= 1.0

    # Registering a new observation invalidates cached GPs, but not history.
    workflow.tell(selected, {"X": selected.intervention.values["X"], "Y": 0.8})
    rebuilt = workflow.fit()
    assert rebuilt[0] is not models[0]
    assert rebuilt[0].train_X.shape[0] == 3
    assert len(workflow.observations) == 3


def test_rejects_invalid_workflow_configuration():
    graph = CausalGraph({"X": (), "Y": ("X",)})
    with pytest.raises(ValueError, match="at least 2"):
        CausalBOStudy(graph, "Y", (("X",),), {"X": (0.0, 1.0)}, n_initial_trials=1)
    with pytest.raises(ValueError, match="non-outcome"):
        CausalBOStudy(graph, "Y", (("Y",),), {"Y": (0.0, 1.0)})
    with pytest.raises(ValueError, match="positive"):
        CausalBOStudy(
            graph, "Y", (("X",),), {"X": (0.0, 1.0)},
            intervention_costs={frozenset({"X"}): 0.0},
        )
