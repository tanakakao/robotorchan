"""Validate multiobjective empirical cohort inputs and immutable artifacts."""

import pytest

from robotorchan.benchmarks.empirical_multiobjective import run_multiobjective


@pytest.mark.parametrize(
    ("problem", "q", "mc_samples"),
    [
        ("branin", 1, 128),
        ("branin_currin", 2, 128),
        ("branin_currin", 1, 0),
    ],
)
def test_multiobjective_rejects_invalid_cohorts(tmp_path, problem, q, mc_samples):
    with pytest.raises(ValueError):
        run_multiobjective(
            tmp_path / "invalid",
            problem=problem,
            q=q,
            git_sha="abc123",
            dirty=False,
            mc_samples=mc_samples,
        )


def test_multiobjective_refuses_overwrite(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_multiobjective(
            destination,
            problem="branin_currin",
            q=1,
            git_sha="abc123",
            dirty=False,
        )
