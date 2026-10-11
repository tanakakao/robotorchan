"""Validation of the empirical noiseless execution entry point."""

import pytest

from robotorchan.benchmarks.empirical_noiseless import run_noiseless_singleobjective


@pytest.mark.parametrize(
    ("problem", "q"),
    [("unknown", 1), ("branin", 2), ("hartmann6", 0)],
)
def test_empirical_noiseless_rejects_invalid_cohorts(tmp_path, problem, q):
    with pytest.raises(ValueError):
        run_noiseless_singleobjective(
            tmp_path / "invalid",
            problem=problem,
            q=q,
            git_sha="abc123",
            dirty=False,
        )


def test_empirical_noiseless_requires_source_provenance(tmp_path):
    with pytest.raises(ValueError, match="git SHA"):
        run_noiseless_singleobjective(
            tmp_path / "missing",
            problem="branin",
            q=1,
            git_sha="",
            dirty=False,
        )


def test_empirical_noiseless_refuses_existing_artifacts(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_noiseless_singleobjective(
            destination,
            problem="branin",
            q=1,
            git_sha="abc123",
            dirty=False,
        )


@pytest.mark.parametrize(
    "strategies",
    [
        (),
        ("random", "random", "sobol"),
        ("random", "sobol", "not_a_strategy"),
        ("qEI", "sobol"),
        ("random", "qEI"),
    ],
)
def test_empirical_noiseless_rejects_invalid_strategy_cohorts(tmp_path, strategies):
    with pytest.raises(ValueError):
        run_noiseless_singleobjective(
            tmp_path / "invalid_strategies",
            problem="branin",
            q=1,
            git_sha="abc123",
            dirty=False,
            strategies=strategies,
        )


def test_empirical_noiseless_extended_strategy_registry():
    from robotorchan.benchmarks.empirical_noiseless import (
        EXTENDED_STRATEGIES,
        RUNNERS,
        STRATEGIES,
    )

    assert STRATEGIES == ("random", "sobol", "qEI", "qNEI")
    assert EXTENDED_STRATEGIES == ("qLogEI", "qLogNEI", "qPI", "qUCB", "qTS")
    assert set(STRATEGIES + EXTENDED_STRATEGIES) == set(RUNNERS)
