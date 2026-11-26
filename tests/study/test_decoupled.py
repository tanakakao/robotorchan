"""Named partial observations in the existing Study lifecycle."""

import pytest
import torch
from botorch.models import ModelListGP, SingleTaskGP

from robotorchan.acquisition import qDecoupledExpectedPredictiveInformationGain
from robotorchan.study import create_decoupled_proposer, create_study
from robotorchan.study.trial import TrialState


def proposer(history, bounds, *, q, fixed_features):
    points = bounds.mean(dim=0).expand(q, -1)
    mask = torch.ones(q, 2, dtype=torch.bool)
    return points, mask, torch.tensor(0.0)


def test_partial_observations_resume_and_failure():
    kwargs = {
        "search_space": {"x": (0.0, 1.0)},
        "output_names": ("strength", "quality"),
        "decoupled_proposer": proposer,
    }
    study = create_study(**kwargs)
    a, b = study.ask_batch(2)
    partial = study.tell(a, observations={"strength": 2.0})
    assert partial.state is TrialState.RUNNING
    assert a.observations == {}
    with pytest.raises(ValueError, match="already"):
        study.tell(a, observations={"strength": 3.0})
    study.tell(b, observations={"quality": 1.0}, state=TrialState.FAIL)
    restored = create_study(**kwargs, initial_trials=study.trials)
    done = restored.tell(a, observations={"quality": 0.0})
    assert done.state is TrialState.COMPLETE
    assert done.observations == {"strength": 2.0, "quality": 0.0}
    assert restored.get_trial(b.number).observations == {"quality": 1.0}
    assert restored.ask().number == 2
    with pytest.raises(ValueError, match="scalar"):
        _ = restored.best_trial


def test_real_gp_proposer_uses_only_observed_outputs():
    seen = []
    X = torch.linspace(0, 1, 5, dtype=torch.double).unsqueeze(-1)

    def model_factory(data):
        seen.append(data)
        models = []
        for j, (train_X, train_Y) in enumerate(data.values()):
            model = SingleTaskGP(
                torch.cat([X, train_X]),
                torch.cat([X.sin() + j, train_Y]),
            )
            model.eval()
            models.append(model)
        return ModelListGP(*models)

    masks = torch.tensor([[True, False], [False, True]])

    def acquisition_factory(model):
        return qDecoupledExpectedPredictiveInformationGain(
            model=model,
            target_X=X,
            X_evaluation_mask=masks[:1],
            observation_noise=X.new_tensor([0.1, 0.1]),
        )

    propose = create_decoupled_proposer(
        parameter_names=("x",),
        output_names=("a", "b"),
        evaluation_masks=masks,
        model_factory=model_factory,
        acquisition_factory=acquisition_factory,
        optimizer_kwargs={"optimizer": "sobol", "optimizer_options": {"num_samples": 8}},
    )
    study = create_study(
        search_space={"x": (0.0, 1.0)},
        output_names=("a", "b"),
        decoupled_proposer=propose,
    )
    trial = study.ask()
    with pytest.raises(ValueError, match="Pending"):
        study.ask()
    name = trial.evaluation_mask[0]
    study.tell(trial, observations={name: 0.5})
    study.ask()
    assert seen[-1][name][1].shape == (1, 1)
    other = "b" if name == "a" else "a"
    assert seen[-1][other][1].shape == (0, 1)


def test_retry_preserves_partial_results():
    study = create_study(
        search_space={"x": (0.0, 1.0)},
        output_names=("strength", "quality"),
        decoupled_proposer=proposer,
    )
    trial = study.ask()
    study.tell(trial, observations={"strength": 2.0}, state=TrialState.FAIL)
    retry = study.retry_failed(trial.number)
    assert retry.params == trial.params
    assert retry.evaluation_mask == ("quality",)
    assert retry.number != trial.number
    study.tell(retry, observations={"quality": 1.0})
    assert study.get_trial(trial.number).observations == {"strength": 2.0}


def test_fixed_cost_budget_from_history(monkeypatch):
    from robotorchan.study import decoupled

    def optimize(acq, bounds, masks, **kwargs):
        assert kwargs["budget"].remaining == 2.0
        torch.testing.assert_close(kwargs["output_costs"], bounds.new_tensor([1.0, 2.0]))
        return bounds.mean(dim=0).unsqueeze(0), masks[:1], bounds.new_tensor(0.0)

    monkeypatch.setattr(decoupled, "optimize_acqf_decoupled", optimize)
    propose = create_decoupled_proposer(
        parameter_names=("x",),
        output_names=("a", "b"),
        evaluation_masks=torch.tensor([[True, False]]),
        model_factory=lambda data: None,
        acquisition_factory=lambda model: None,
        allow_pending=True,
        output_costs={"a": 1.0, "b": 2.0},
        total_budget=5.0,
    )
    study = create_study(
        search_space={"x": (0.0, 1.0)},
        output_names=("a", "b"),
        decoupled_proposer=propose,
    )
    trial = study.add_trial({"x": 0.5}, evaluation_mask=("a", "b"))
    study.start_trial(trial.number)
    study.tell(trial, observations={"a": 1.0})
    study.ask()
