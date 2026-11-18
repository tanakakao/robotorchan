"""Focused contextual Study ask/tell contract tests."""

import pytest

from robotorchan.study import create_study


def test_contextual_study_initial_ask_tell_preserves_observed_context() -> None:
    study = create_study(
        search_space={"decision": (0.0, 1.0), "ambient": (0.0, 1.0)},
        context_names=("ambient",),
        n_initial_trials=3,
        seed=12,
    )
    first = study.ask(context={"ambient": 0.25})
    assert first.params["ambient"] == pytest.approx(0.25)
    study.tell(first, 0.4)
    second = study.ask(context={"ambient": 0.75})
    assert second.params["ambient"] == pytest.approx(0.75)
    assert len(study.trials) == 2


def test_contextual_study_requires_complete_context() -> None:
    study = create_study(
        search_space={"decision": (0.0, 1.0), "ambient": (0.0, 1.0)},
        context_names=("ambient",),
    )
    with pytest.raises(ValueError, match="context"):
        study.ask()
    with pytest.raises(ValueError, match="context"):
        study.ask(context={"decision": 0.2})
    with pytest.raises(ValueError, match="outside"):
        study.ask(context={"ambient": 2.0})


def test_regular_study_rejects_unconfigured_context() -> None:
    study = create_study(search_space={"decision": (0.0, 1.0)})
    with pytest.raises(ValueError, match="no configured"):
        study.ask(context={"decision": 0.5})
