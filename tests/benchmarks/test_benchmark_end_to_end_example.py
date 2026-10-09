"""Smoke test for the public benchmark example."""

import runpy
from pathlib import Path


def test_example_creates_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    example = Path(__file__).resolve().parents[2] / "examples" / "benchmark_end_to_end.py"
    runpy.run_path(str(example), run_name="__main__")
    directory = tmp_path / "benchmark_outputs"
    assert (directory / "branin_random_results.json").is_file()
    assert (directory / "branin_random_report.json").is_file()
    assert "simple_regret" in (directory / "branin_random_report.csv").read_text(encoding="utf-8")
