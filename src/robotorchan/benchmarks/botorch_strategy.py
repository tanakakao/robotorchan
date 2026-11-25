"""BoTorch-native single-objective strategy for the benchmark runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from inspect import signature
from typing import Any

import torch
from botorch.acquisition import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from robotorchan.acquisition.factory import create_acquisition, get_acquisition_class
from robotorchan.benchmarks.problem import BenchmarkProblem
from robotorchan.optim.capabilities import OPTIMIZER_CAPABILITIES
from robotorchan.optim.factory import create_optimizer

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
    model_factory: ModelFactory | str = SingleTaskGP,
    model_kwargs: Mapping[str, Any] | None = None,
    acquisition_kwargs: Mapping[str, Any] | None = None,
    optimizer: str | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
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
    legacy_acquisitions = {
        "qEI": qExpectedImprovement,
        "qLogEI": qLogExpectedImprovement,
        "qNEI": qNoisyExpectedImprovement,
        "qLogNEI": qLogNoisyExpectedImprovement,
        "qPI": qProbabilityOfImprovement,
        "qUCB": qUpperConfidenceBound,
    }
    constructor = legacy_acquisitions.get(acquisition)
    if constructor is None:
        try:
            constructor = get_acquisition_class(acquisition)
        except KeyError as error:
            raise ValueError(f"Unknown acquisition: {acquisition}") from error
    if optimizer is not None and optimizer.lower() not in OPTIMIZER_CAPABILITIES:
        raise ValueError("Benchmark requires a scalar dispatcher optimizer.")
    if optimizer is None and optimizer_kwargs:
        raise ValueError("optimizer_kwargs requires an explicit optimizer name.")
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
    seed = int(torch.randint(0, 2**31 - 1, (1,), device=X.device, generator=generator).item())
    devices = [X.device.index or 0] if X.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if isinstance(model_factory, str):
            model = import_module("robotorchan.models").create_model(
                model_factory, X, train_Y, **dict(model_kwargs or {})
            )
            fit = getattr(model, "fit", None)
            if not callable(fit):
                raise TypeError("Named benchmark model must expose fit().")
            fit()
        else:
            model = model_factory(X, train_Y, **dict(model_kwargs or {}))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        kwargs = dict(acquisition_kwargs or {})
        parameters = signature(constructor).parameters
        if "best_f" in parameters:
            kwargs.setdefault("best_f", train_Y.max())
        if "X_baseline" in parameters:
            kwargs.setdefault("X_baseline", X)
        if constructor is qUpperConfidenceBound:
            kwargs.setdefault("beta", 0.2)
        acqf = create_acquisition(constructor, model=model, **kwargs)
        if optimizer is None:
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
        else:
            options = dict(optimizer_kwargs or {})
            options.setdefault("num_restarts", num_restarts)
            options.setdefault("raw_samples", raw_samples)
            optimize = create_optimizer(optimizer, **options)
            candidates, _ = optimize(acqf, bounds, q=q)
    return candidates.detach()


def make_botorch_gp_strategy(
    *,
    acquisition: str = "qNEI",
    num_restarts: int = 3,
    raw_samples: int = 64,
    model_factory: ModelFactory | str = SingleTaskGP,
    model_kwargs: Mapping[str, Any] | None = None,
    acquisition_kwargs: Mapping[str, Any] | None = None,
    optimizer: str | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
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
            model_kwargs=model_kwargs,
            acquisition_kwargs=acquisition_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

    return propose
