"""Run the comparison entry point without optional external packages."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from robotorchan.benchmarks.persistence import trajectory_from_record

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/cross_library/run.py"


def test_shared_fixture_deterministic_trajectory_and_budget(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            [
                dict(problem="noisy_quadratic", library=name, q=2, initial=4, total=9, mode="A")
                for name in ("random", "sobol")
            ]
        ),
        encoding="utf-8",
    )

    def run(name: str) -> list[dict]:
        target = tmp_path / name
        csv_target = target.with_suffix(".csv")
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--config",
                str(config),
                "--output",
                str(target),
                "--summary",
                str(csv_target),
            ],
            check=True,
            timeout=90,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        with csv_target.open(encoding="utf-8", newline="") as summary:
            assert [row["status"] for row in csv.DictReader(summary)] == ["ok", "ok"]
        return records

    first, second = run("first.jsonl"), run("second.jsonl")
    assert first[0]["initial_data_id"] == first[1]["initial_data_id"]
    assert first[0]["initial_data"] == first[1]["initial_data"]
    candidate_histories = []
    for a, b in zip(first, second, strict=True):
        assert a["status"] == b["status"] == "ok"
        assert a["trajectory"]["tensors"] == b["trajectory"]["tensors"]
        assert a["metrics"] == b["metrics"]
        trajectory = trajectory_from_record(a["trajectory"])
        assert a["evaluation_count"] == 9
        assert trajectory.completed_batches == (2, 4, 5)
        assert trajectory.evaluation_count == 5
        assert len(a["metrics"]["simple_regret"]) == 9
        candidate_histories.append(trajectory.X[4:].tolist())
    assert candidate_histories[0] != candidate_histories[1]


def test_unsupported_case_writes_a_failure_record(tmp_path: Path) -> None:
    config, output = tmp_path / "config.json", tmp_path / "output.jsonl"
    config.write_text(
        json.dumps(
            [
                dict(
                    problem="constrained_annulus",
                    library="baybe",
                    q=1,
                    initial=4,
                    total=5,
                    mode="A",
                )
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config), "--output", str(output)],
        check=False,
        timeout=90,
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert row["status"] == "unsupported"
    assert row["error"]
    assert row["evaluation_count"] == 0
    assert row["trajectory"] is None
