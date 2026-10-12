"""Check that cost-aware benchmarks evaluate target fidelity separately."""

import torch

from robotorchan.benchmarks.multifidelity_problems import multifidelity_quadratic
from robotorchan.benchmarks.runner import sobol_initial_design


def test_multifidelity_target_quality_differs_from_observed_fidelity():
    problem = multifidelity_quadratic()
    X = sobol_initial_design(
        problem, 8, 0, dtype=torch.float64, device=torch.device("cpu")
    )
    target = X.clone()
    target[:, -1] = 1.0
    assert torch.all(problem.evaluate_cost(X) >= 0.1)
    assert torch.all(problem.evaluate_truth(target) <= problem.evaluate_truth(X))
    assert torch.allclose(problem.evaluate_cost(target), torch.ones(8, 1, dtype=X.dtype))
