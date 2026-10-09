"""BoTorch-native qEHVI candidate generation for multiobjective benchmarks."""

from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch.mlls import SumMarginalLogLikelihood
from torch import Size, Tensor

from robotorchan.benchmarks.problem import BenchmarkProblem


def botorch_qehvi_candidates(
    problem: BenchmarkProblem,
    X: Tensor,
    Y: Tensor,
    q: int,
    generator: torch.Generator,
    *,
    num_restarts: int = 3,
    raw_samples: int = 64,
    mc_samples: int = 128,
) -> Tensor:
    """Fit independent objective GPs and optimize native BoTorch qEHVI.

    The surrogate and partitioning use observed outcomes only, never truth.
    """
    if problem.n_objectives != 2 or problem.n_constraints:
        raise ValueError("qEHVI requires unconstrained two-objective problems.")
    if any(kind != "continuous" for kind in problem.variable_types):
        raise ValueError("qEHVI requires continuous variables.")
    if problem.reference_point is None:
        raise ValueError("qEHVI requires a reference point.")
    if type(num_restarts) is not int or num_restarts < 1:
        raise ValueError("num_restarts must be positive.")
    if type(raw_samples) is not int or raw_samples < 1:
        raise ValueError("raw_samples must be positive.")
    if type(mc_samples) is not int or mc_samples < 1:
        raise ValueError("mc_samples must be positive.")
    if q < 1 or X.ndim != 2 or X.shape[1] != problem.dimension:
        raise ValueError("Expected X=(n,d) and q >= 1.")
    if Y.shape != (X.shape[0], 2) or X.shape[0] < 2:
        raise ValueError("Expected Y=(n,2) and at least two observations.")

    signs = Y.new_tensor([1 if d == "maximize" else -1 for d in problem.directions])
    train_Y = Y * signs
    reference = problem.to_maximization(problem.reference_point.to(dtype=Y.dtype, device=Y.device))
    bounds = problem.bounds.to(dtype=X.dtype, device=X.device)
    seed = int(torch.randint(0, 2**31 - 1, (1,), device=X.device, generator=generator).item())
    devices = list(range(torch.cuda.device_count())) if X.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = ModelListGP(*[SingleTaskGP(X, train_Y[:, i : i + 1]) for i in range(2)])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        partitioning = NondominatedPartitioning(ref_point=reference, Y=train_Y)
        acqf = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=reference.tolist(),
            partitioning=partitioning,
            sampler=SobolQMCNormalSampler(sample_shape=Size([mc_samples]), seed=seed),
        )
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    return candidates.detach()
