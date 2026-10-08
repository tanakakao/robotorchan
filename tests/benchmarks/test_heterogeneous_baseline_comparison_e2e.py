"""Comparable fixed-budget BO, random, and Sobol benchmark baselines."""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.composition import make_qei_acquisition
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics import ProblemSemantics, RegressionObjective


def _run_strategy(
    strategy: str,
    initial_X: torch.Tensor,
    *,
    seed: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return evaluated inputs and deterministic best-so-far truth by evaluation count."""
    train_X = initial_X.clone()
    train_Y = observe(train_X, seed=seed + 100).strength.unsqueeze(-1)
    truth, _, _ = evaluate_truth(train_X)
    best = [truth.max()]
    generator = torch.Generator().manual_seed(seed + 200)
    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed + 300)
    bounds = torch.stack((torch.zeros(3, dtype=torch.double), torch.ones(3, dtype=torch.double)))
    semantics = ProblemSemantics(objectives=(RegressionObjective(0),))

    for iteration in range(steps):
        if strategy == "random":
            candidate = torch.rand(1, 3, generator=generator, dtype=torch.double)
        elif strategy == "sobol":
            candidate = sobol.draw(1).to(dtype=torch.double)
        elif strategy == "qei":
            regression = SingleTaskGP(
                train_X,
                train_Y,
                train_Yvar=torch.full_like(train_Y, 0.02**2),
            )
            fit_gpytorch_mll(regression.make_mll())
            regression.eval()
            acquisition = make_qei_acquisition(
                HeterogeneousModel(regression),
                semantics,
                best_f=train_Y.max().item(),
                sampler=SobolQMCNormalSampler(
                    sample_shape=torch.Size([16]),
                    seed=seed + iteration,
                ),
            )
            candidate, _ = optimize_acqf(
                acquisition,
                bounds,
                q=1,
                optimizer="sobol",
                raw_samples=32,
                seed=seed + iteration,
            )
        else:
            raise ValueError(f"Unknown benchmark strategy: {strategy}")

        assert candidate.shape == (1, 3)
        assert torch.isfinite(candidate).all()
        assert ((candidate >= 0) & (candidate <= 1)).all()
        with torch.no_grad():
            new_truth, _, _ = evaluate_truth(candidate)
            new_Y = observe(candidate, seed=seed + 400 + iteration).strength.unsqueeze(-1)
        best.append(torch.maximum(best[-1], new_truth.max()))
        train_X = torch.cat((train_X, candidate), dim=0)
        train_Y = torch.cat((train_Y, new_Y), dim=0)

    return train_X, torch.stack(best)


def test_fixed_budget_random_sobol_and_qei_baselines() -> None:
    """Use shared initial observations and count all new function evaluations."""
    seed = 1414
    initial_X = torch.rand(
        12,
        3,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.double,
    )
    results = {
        strategy: _run_strategy(strategy, initial_X, seed=seed, steps=3)
        for strategy in ("random", "sobol", "qei")
    }
    initial_truth, _, _ = evaluate_truth(initial_X)
    for evaluated_X, best in results.values():
        assert evaluated_X.shape == (15, 3)
        torch.testing.assert_close(evaluated_X[:12], initial_X)
        assert best.shape == (4,)
        assert torch.isfinite(best).all()
        torch.testing.assert_close(best[0], initial_truth.max())
        assert (best[1:] >= best[:-1]).all()
        final_truth, _, _ = evaluate_truth(evaluated_X)
        torch.testing.assert_close(best[-1], final_truth.max())
    assert len(results) == 3
