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
