"""Phase 11 qNEHVI comparative benchmark tests."""

import pytest
import torch

from robotorchan.benchmarks.botorch_multiobjective_strategy import botorch_qehvi_candidates
from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qnehvi import (
    make_comparative_qnehvi_strategy,
    run_comparative_qnehvi,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.metrics import hypervolume_curve
from robotorchan.benchmarks.multiobjective_problems import branin_currin


def _cell(strategy: str = "qNEHVI") -> ComparativeExperimentCell:
    return ComparativeExperimentCell(
        tier="smoke",
        config=BenchmarkExperimentConfig(
            problem="branin_currin",
            strategy=strategy,
            seeds=(0,),
            initial_points=4,
            evaluation_budget=4,
            q=1,
        ),
    )


def test_qnehvi_smoke_reproducible_and_truth_scored() -> None:
    first = run_comparative_qnehvi(_cell(), num_restarts=1, raw_samples=8, mc_samples=16)[0]
    second = run_comparative_qnehvi(_cell(), num_restarts=1, raw_samples=8, mc_samples=16)[0]
    assert first.evaluation_count == 4
    assert first.Y_truth.shape == (8, 2)
    torch.testing.assert_close(first.X, second.X)
    hv = hypervolume_curve(branin_currin(), first)
    assert torch.isfinite(hv).all()
    assert (hv[1:] + 1e-8 >= hv[:-1]).all()


def test_qnehvi_rejects_other_strategy() -> None:
    with pytest.raises(ValueError, match="qNEHVI cell"):
        run_comparative_qnehvi(_cell("random"))


@pytest.mark.parametrize("name", ["num_restarts", "raw_samples", "mc_samples"])
def test_qnehvi_rejects_invalid_optimizer_settings(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        make_comparative_qnehvi_strategy(**{name: 0})


def test_qnehvi_rejects_unknown_acquisition() -> None:
    problem = branin_currin()
    X = torch.rand(4, 2, dtype=torch.double)
    Y = problem.evaluate(X)
    with pytest.raises(ValueError, match="acquisition"):
        botorch_qehvi_candidates(
            problem,
            X,
            Y,
            1,
            torch.Generator().manual_seed(0),
            acquisition="unknown",
        )
