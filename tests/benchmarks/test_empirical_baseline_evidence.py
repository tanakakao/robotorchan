"""Exercise the measured Random/Sobol baseline evidence path."""

import json

import pytest

from robotorchan.benchmarks.empirical_baseline_evidence import run_baseline_cohort


@pytest.mark.parametrize("q, expected_batches", [(1, 50), (3, 17)])
def test_real_baseline_cohort(tmp_path, q, expected_batches):
    """Generate actual ten-seed Branin evidence with paired initial designs."""
    destination = tmp_path / "baseline"
    run_baseline_cohort(destination, problem="branin", q=q, git_sha="test-commit")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "runtime_validated"
    assert manifest["pairing"] == {"design": True, "truth": True}
    assert len(list(destination.glob("*_seed*.json"))) == 20
    assert (destination / "baseline.csv").exists()
    for strategy in ("random", "sobol"):
        record = json.loads((destination / f"{strategy}_seed0.json").read_text(encoding="utf-8"))
        assert len(record["completed_batches"]) == expected_batches
        assert record["completed_batches"][-1] == 50


def test_baseline_rejects_existing_destination(tmp_path):
    destination = tmp_path / "baseline"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        run_baseline_cohort(destination, problem="branin", q=1, git_sha="test-commit")
