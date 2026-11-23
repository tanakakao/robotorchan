"""Phase 5: reuse CausalBOStudy for GP-SCM and native MC acquisition dispatch."""

from __future__ import annotations

from math import isclose, isfinite, log

import pytest
import torch

from robotorchan.causal import (
    CausalAcquisitionProposal,
    CausalBOStudy,
    CausalGraph,
    CausalObservation,
    GPStructuralCausalModel,
    HardIntervention,
    propose_gp_scm_intervention,
)
from robotorchan.causal.gp_scm_proposal import _GPScmMCLogEI
from robotorchan.study import TrialState


@pytest.fixture(scope="module")
def graph_and_observations():
    graph = CausalGraph({"X": (), "Z": ("X",), "Y": ("Z",)})
    observations = [
        CausalObservation({"X": x, "Z": 1.5 * x, "Y": 0.3 + 1.5 * x})
        for x in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    return graph, observations


@pytest.fixture(scope="module")
def fitted_model(graph_and_observations):
    graph, observations = graph_and_observations
    scm = GPStructuralCausalModel(graph, observations)
    scm.fit()
    return scm


def test_standalone_native_mc_proposal_and_determinism(fitted_model):
    result = propose_gp_scm_intervention(
        fitted_model,
        (("X",), ("Z",)),
        {"X": (0.0, 1.0), "Z": (0.0, 1.5)},
        outcome="Y",
        best_f=0.5,
        optimizer="sobol",
        raw_samples=12,
        mc_samples=32,
    )
    assert isinstance(result, CausalAcquisitionProposal)
    assert set(result.intervention.values) in ({"X"}, {"Z"})
    assert isfinite(result.acquisition_value)
    assert isfinite(result.ranking_score)

    acquisition = _GPScmMCLogEI(
        fitted_model,
        nodes=("X",),
        outcome="Y",
        best_f=0.5,
        maximize=True,
        mc_samples=32,
        mc_seed=23,
    )
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double, requires_grad=True)
    torch.manual_seed(101)
    prior = torch.random.get_rng_state()
    first = acquisition(X)
    second = acquisition(X)
    assert first.shape == (2,)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(torch.random.get_rng_state(), prior)
    first.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


def test_same_native_value_is_ranked_by_set_cost(fitted_model, monkeypatch):
    def fake_optimize(acqf, bounds, q, **kwargs):
        assert isinstance(acqf, _GPScmMCLogEI)
        assert q == 1
        return bounds.mean(0, keepdim=True), torch.tensor(-1.0)

    monkeypatch.setattr("robotorchan.causal.proposal.optimize_acqf", fake_optimize)
    result = propose_gp_scm_intervention(
        fitted_model,
        [("X",), ("Z",)],
        {"X": (0.0, 1.0), "Z": (0.0, 1.5)},
        outcome="Y",
        best_f=1.0,
        direction="minimize",
        intervention_costs={frozenset({"X"}): 10.0, frozenset({"Z"}): 2.0},
    )
    assert set(result.intervention.values) == {"Z"}
    assert result.acquisition_value == -1.0
    assert isclose(result.ranking_score, -1.0 - log(2.0))


def test_study_uses_observational_data_and_one_initial_intervention(graph_and_observations):
    graph, observations = graph_and_observations
    study = CausalBOStudy(
        graph,
        "Y",
        (("X",), ("Z",)),
        {"X": (0.0, 1.0), "Z": (0.0, 1.5)},
        model_type="gp_scm",
        observational_data=observations,
        n_initial_trials=5,  # GP-SCM does not require five per intervention set.
        optimizer="sobol",
        raw_samples=12,
        mc_samples=32,
        seed=9,
    )
    assert not study.observations
    prior_model = study.fit()
    assert isinstance(prior_model, GPStructuralCausalModel)
    assert study.fit() is prior_model
    assert len(prior_model.mechanisms) == 2

    first = study.ask()
    assert first.state is TrialState.RUNNING
    assert first.proposal is None
    assert set(first.intervention.values) == {"X"}
    with pytest.raises(RuntimeError, match="running"):
        study.ask()

    x = first.intervention.values["X"]
    complete = study.tell(first, {"Z": 1.5 * x, "Y": 0.3 + 1.5 * x})
    assert complete.state is TrialState.COMPLETE
    assert complete.observation is not None
    assert complete.observation.intervention == first.intervention
    assert len(study.observations) == 1
    assert len(study.observational_data) == 6
    assert study.fit() is not prior_model

    next_trial = study.ask()
    assert next_trial.state is TrialState.RUNNING
    assert isinstance(next_trial.proposal, CausalAcquisitionProposal)
    assert set(next_trial.intervention.values) in ({"X"}, {"Z"})
    cached = study.fit()

    failed = study.tell(next_trial, state=TrialState.FAIL)
    assert failed.state is TrialState.FAIL
    assert study.fit() is cached
    assert len(study.observations) == 1
    retry = study.ask()
    assert retry.number == 2
    assert isinstance(retry.proposal, CausalAcquisitionProposal)
    with pytest.raises(ValueError, match="RUNNING"):
        study.tell(next_trial, {"Y": 0.8})


def test_insufficient_full_nodes_are_explicit_not_imputed(graph_and_observations):
    graph, _ = graph_and_observations
    study = CausalBOStudy(
        graph,
        "Y",
        (("X",),),
        {"X": (0.0, 1.0)},
        model_type="gp_scm",
        optimizer="sobol",
        raw_samples=8,
    )
    first = study.ask()
    study.tell(first, 0.4)
    assert len(study.observations) == 1
    with pytest.raises(ValueError, match=r"Node 'X'.*two complete"):
        study.fit()
    with pytest.raises(ValueError, match="two complete"):
        study.ask()
    # A failed model-based ask() must not issue a new RUNNING trial.
    assert len(study.trials) == 1


def test_failed_initial_trial_retries_without_adding_observations(graph_and_observations):
    graph, observations = graph_and_observations
    study = CausalBOStudy(
        graph,
        "Y",
        (("X",),),
        {"X": (0.0, 1.0)},
        model_type="gp_scm",
        observational_data=observations,
        seed=5,
    )
    first = study.ask()
    study.tell(first, state=TrialState.FAIL)
    assert not study.observations
    again = study.ask()
    assert again.number == 1
    assert again.proposal is None


def test_rejects_invalid_gpscm_setup(graph_and_observations, fitted_model):
    graph, observations = graph_and_observations
    args = (graph, "Y", (("X",),), {"X": (0.0, 1.0)})
    with pytest.raises(ValueError, match="model_type"):
        CausalBOStudy(*args, model_type="unknown")
    with pytest.raises(ValueError, match="only used"):
        CausalBOStudy(*args, observational_data=observations)
    with pytest.raises(ValueError, match="not contain do-interventions"):
        CausalBOStudy(
            *args,
            model_type="gp_scm",
            observational_data=[
                CausalObservation(
                    {"X": 0.1, "Z": 0.15, "Y": 0.45},
                    HardIntervention({"X": 0.1}),
                ),
            ],
        )
    with pytest.raises(ValueError, match="mc_samples"):
        CausalBOStudy(*args, model_type="gp_scm", mc_samples=0)
    with pytest.raises(ValueError, match="mc_samples"):
        CausalBOStudy(*args, model_type="gp_scm", mc_seed=-1)

    with pytest.raises(ValueError, match="only once"):
        propose_gp_scm_intervention(
            fitted_model,
            [("X", "Z"), ("Z", "X")],
            {"X": (0, 1), "Z": (0, 1)},
            outcome="Y",
            best_f=1.0,
        )
    with pytest.raises(ValueError, match="positive finite"):
        propose_gp_scm_intervention(
            fitted_model,
            [("X",)],
            {"X": (0, 1)},
            outcome="Y",
            best_f=1.0,
            intervention_costs={frozenset({"X"}): 0.0},
        )
    with pytest.raises(ValueError, match="finite interventional"):
        propose_gp_scm_intervention(
            fitted_model,
            [("X",)],
            {"X": (0, 1)},
            outcome="Y",
            best_f=float("nan"),
        )
    with pytest.raises(ValueError, match="direction"):
        propose_gp_scm_intervention(
            fitted_model,
            [("X",)],
            {"X": (0, 1)},
            outcome="Y",
            best_f=0.5,
            direction="invalid",
        )



def test_gp_scm_initialization_collects_missing_mechanism_records():
    """Do-only Y measurements can bootstrap a GP if root X is observed already."""
    graph = CausalGraph({"X": (), "Y": ("X",)})
    root_data = [
        CausalObservation({"X": 0.1}),
        CausalObservation({"X": 0.9}),
    ]
    study = CausalBOStudy(
        graph,
        "Y",
        (("X",),),
        {"X": (0.0, 1.0)},
        model_type="gp_scm",
        observational_data=root_data,
        n_initial_trials=1,
        optimizer="sobol",
        raw_samples=8,
        mc_samples=16,
    )
    for step in range(2):
        trial = study.ask()
        assert trial.proposal is None
        assert trial.number == step
        x = trial.intervention.values["X"]
        study.tell(trial, {"Y": 1.0 + x})
    assert len(study.observations) == 2
    assert isinstance(study.fit(), GPStructuralCausalModel)

    proposed = study.ask()
    assert isinstance(proposed.proposal, CausalAcquisitionProposal)
    assert 0.0 <= proposed.intervention.values["X"] <= 1.0


def test_gp_scm_reports_unidentifiable_root_instead_of_repeating_sobol():
    """A root manipulated by every available intervention needs observational data."""
    graph = CausalGraph({"X": (), "Y": ("X",)})
    study = CausalBOStudy(
        graph,
        "Y",
        (("X",),),
        {"X": (0.0, 1.0)},
        model_type="gp_scm",
    )
    first = study.ask()
    study.tell(first, {"Y": 0.7})
    with pytest.raises(ValueError, match="observational_data"):
        study.ask()
    assert len(study.trials) == 1


def test_gp_scm_requests_missing_intermediate_measurements_in_tell():
    """Do not pretend unmeasured mediators were observed."""
    graph = CausalGraph({"X": (), "Z": ("X",), "Y": ("Z",)})
    study = CausalBOStudy(
        graph,
        "Y",
        (("X",),),
        {"X": (0.0, 1.0)},
        model_type="gp_scm",
        observational_data=[
            CausalObservation({"X": 0.1}),
            CausalObservation({"X": 0.9}),
        ],
    )
    first = study.ask()
    study.tell(first, {"Y": 0.7})  # Z is unknown, not an inferred measurement
    with pytest.raises(ValueError, match=r"tell\\(\\)"):
        study.ask()
    assert len(study.trials) == 1


def test_constant_root_mc_logei_matches_exact_improvement_and_cost():
    """Known root-only do model has an analytic EI for both directions."""
    graph = CausalGraph({"X": (), "Y": ()})
    model = GPStructuralCausalModel(
        graph,
        [
            CausalObservation({"X": 0.1, "Y": 1.0}),
            CausalObservation({"X": 0.9, "Y": 1.0}),
        ],
    )
    model.fit()
    X = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
    for maximize, incumbent in ((True, 0.75), (False, 1.25)):
        acquisition = _GPScmMCLogEI(
            model,
            nodes=("X",),
            outcome="Y",
            best_f=incumbent,
            maximize=maximize,
            mc_samples=32,
            mc_seed=11,
        )
        native_values = acquisition(X)
        torch.testing.assert_close(
            native_values,
            torch.full((2,), log(0.25), dtype=torch.double),
        )
    selected = propose_gp_scm_intervention(
        model,
        [("X",)],
        {"X": (0.0, 1.0)},
        outcome="Y",
        best_f=0.75,
        optimizer="sobol",
        raw_samples=8,
        intervention_costs={frozenset({"X"}): 2.0},
        mc_samples=16,
    )
    assert isclose(selected.acquisition_value, log(0.25), abs_tol=1e-10)
    assert isclose(selected.ranking_score, log(0.125), abs_tol=1e-10)
