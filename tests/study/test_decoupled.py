"""Named partial observations in the existing Study lifecycle."""

import pytest
import torch

from robotorchan.study import create_study
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
