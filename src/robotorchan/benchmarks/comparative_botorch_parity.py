"""Phase 20 native BoTorch candidate-generation parity checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from robotorchan.benchmarks.botorch_strategy import botorch_gp_candidates
from robotorchan.benchmarks.comparative_qei import make_comparative_qei_strategy
from robotorchan.benchmarks.comparative_qnei import make_comparative_qnei_strategy
from robotorchan.benchmarks.problem import BenchmarkProblem


@dataclass(frozen=True)
class CandidateParity:
    """Same-history, same-RNG native BoTorch candidate comparison."""

    acquisition: str
    reference_candidates: Tensor
    benchmark_candidates: Tensor
    max_absolute_difference: float
    within_tolerance: bool


def compare_botorch_candidates(
    problem: BenchmarkProblem,
    X: Tensor,
    Y: Tensor,
    *,
    acquisition: str,
    q: int = 1,
    seed: int = 0,
    num_restarts: int = 2,
    raw_samples: int = 32,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> CandidateParity:
    """Compare benchmark adapters with native BoTorch under identical RNG state.

    This is an adapter-equivalence check, not an independent optimizer
    performance comparison: the adapters delegate to the native implementation.
    """
    if acquisition not in ("qEI", "qNEI"):
        raise ValueError("Only qEI and qNEI are supported.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if q < 1 or atol < 0 or rtol < 0:
        raise ValueError("q and tolerances must be valid.")
    if X.ndim != 2 or Y.shape != (X.shape[0], 1):
        raise ValueError("Expected X=(n,d) and Y=(n,1).")
    if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
        raise ValueError("Training data must be finite.")

    options = {"num_restarts": num_restarts, "raw_samples": raw_samples}
    adapter = (
        make_comparative_qei_strategy(**options)
        if acquisition == "qEI"
        else make_comparative_qnei_strategy(**options)
    )
    reference_rng = torch.Generator(device=X.device).manual_seed(seed)
    benchmark_rng = torch.Generator(device=X.device).manual_seed(seed)
    reference = botorch_gp_candidates(
        problem, X, Y, q, reference_rng, acquisition=acquisition, **options
    )
    benchmark = adapter(problem, X, Y, q, benchmark_rng)
    if reference.shape != benchmark.shape or not torch.isfinite(benchmark).all():
        raise ValueError("Candidate shapes or values differ from native BoTorch.")
    difference = (reference - benchmark).abs().max().item()
    return CandidateParity(
        acquisition=acquisition,
        reference_candidates=reference,
        benchmark_candidates=benchmark,
        max_absolute_difference=float(difference),
        within_tolerance=bool(torch.allclose(reference, benchmark, atol=atol, rtol=rtol)),
    )
