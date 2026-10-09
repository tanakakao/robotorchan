"""Phase 10 qEHVI benchmark validation."""

import pytest
import torch

from robotorchan.benchmarks.comparative_matrix import ComparativeExperimentCell
from robotorchan.benchmarks.comparative_qehvi import (
    make_comparative_qehvi_strategy,
    run_comparative_qehvi,
)
from robotorchan.benchmarks.config import BenchmarkExperimentConfig
from robotorchan.benchmarks.metrics import hypervolume_curve
from robotorchan.benchmarks.multiobjective_problems import branin_currin


def _cell(strategy: str = "qEHVI") -> ComparativeExperimentCell:
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


def test_qehvi_smoke_and_reproducibility() -> None:
    first = run_comparative_qehvi(_cell(), num_restarts=1, raw_samples=8)[0]
    second = run_comparative_qehvi(_cell(), num_restarts=1, raw_samples=8)[0]
    assert first.evaluation_count == 4
    assert first.Y_truth.shape == (8, 2)
    torch.testing.assert_close(first.X, second.X)
    hv = hypervolume_curve(branin_currin(), first)
    assert torch.isfinite(hv).all()
    assert (hv[1:] + 1e-8 >= hv[:-1]).all()


def test_qehvi_rejects_baseline_strategy() -> None:
    with pytest.raises(ValueError, match="qEHVI cell"):
        run_comparative_qehvi(_cell("random"))


def test_qehvi_validates_optimizer_settings() -> None:
    with pytest.raises(ValueError, match="num_restarts"):
        make_comparative_qehvi_strategy(num_restarts=0)
    with pytest.raises(ValueError, match="raw_samples"):
        make_comparative_qehvi_strategy(raw_samples=0)
    with pytest.raises(ValueError, match="mc_samples"):
        make_comparative_qehvi_strategy(mc_samples=0)
