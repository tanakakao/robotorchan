"""BoTorch-native single-objective strategy for the benchmark runner."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem

ModelFactory = Callable[[Tensor, Tensor], SingleTaskGP]


def botorch_gp_candidates(
    problem: BenchmarkProblem,
    X: Tensor,
    Y: Tensor,
    q: int,
    generator: torch.Generator,
    *,
    acquisition: str = "qNEI",
    num_restarts: int = 3,
    raw_samples: int = 64,
    model_factory: ModelFactory = SingleTaskGP,
) -> Tensor:
    """Fit a GP and optimize qEI or qNEI using native BoTorch APIs.

    This baseline supports unconstrained, continuous, single-objective problems.
    It does not use ground-truth objectives, constraints, or optimal values
    for fitting or acquisition optimization.
    """
    if problem.n_objectives != 1 or problem.n_constraints:
        raise ValueError("GP baseline requires unconstrained single-objective problems.")
    if any(kind != "continuous" for kind in problem.variable_types):
        raise ValueError("GP baseline requires continuous variables.")
    if acquisition not in ("qEI", "qNEI"):
        raise ValueError("acquisition must be qEI or qNEI.")
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")
    if q < 1 or X.ndim != 2 or Y.shape != (X.shape[0], 1):
        raise ValueError("Expected X=(n,d), Y=(n,1), and q >= 1.")
    signs = Y.new_tensor([1 if problem.directions[0] == "maximize" else -1])
    train_Y = Y * signs
    bounds = problem.bounds.to(dtype=X.dtype, device=X.device)
    # Seed the entire fit/acquisition/optimization cycle, not only initial conditions.
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    devices = [X.device.index or 0] if X.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = model_factory(X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        if acquisition == "qEI":
            acqf = qExpectedImprovement(model=model, best_f=train_Y.max())
        else:
            acqf = qNoisyExpectedImprovement(model=model, X_baseline=X)
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    return candidates.detach()


def make_botorch_gp_strategy(
    *,
    acquisition: str = "qNEI",
    num_restarts: int = 3,
    raw_samples: int = 64,
    model_factory: ModelFactory = SingleTaskGP,
) -> Callable[[BenchmarkProblem, Tensor, Tensor, int, torch.Generator], Tensor]:
    """Return a candidate callback compatible with run_benchmark."""

    def propose(
        problem: BenchmarkProblem,
        X: Tensor,
        Y: Tensor,
        q: int,
        generator: torch.Generator,
    ) -> Tensor:
        return botorch_gp_candidates(
            problem,
            X,
            Y,
            q,
            generator,
            acquisition=acquisition,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            model_factory=model_factory,
        )

    return propose
