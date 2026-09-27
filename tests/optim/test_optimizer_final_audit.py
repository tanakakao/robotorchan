"""Phase 20 regression tests for the optimizer final audit."""

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim.backends.mixed_genetic_algorithm import optimize_acqf_mixed_ga
from robotorchan.optim.backends.sampling import optimize_acqf_sampling
from robotorchan.optim.constraint_evaluation import candidate_constraint_violation


class _SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.sum(dim=(-2, -1))


def test_sampling_fixed_features_are_applied_before_scoring() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    candidate, _ = optimize_acqf_sampling(
        _SumAcquisition(),
        bounds,
        q=1,
        num_samples=32,
        method="random",
        seed=1,
        fixed_features={0: 0.25},
    )
    assert candidate[0, 0].item() == 0.25


def test_negative_equality_tolerance_is_rejected() -> None:
    candidates = torch.zeros(1, 1, 1)
    try:
        candidate_constraint_violation(candidates, None, equality_tolerance=-1.0)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Negative equality tolerance must be rejected.")


def test_mixed_ga_integer_result_is_legal_for_fractional_bounds() -> None:
    bounds = torch.tensor([[0.2], [2.8]])
    candidate, _ = optimize_acqf_mixed_ga(
        _SumAcquisition(),
        bounds,
        q=1,
        integer_dims=(0,),
        population_size=16,
        generations=3,
        seed=2,
    )
    assert candidate.item() in {1.0, 2.0}


def test_mixed_ga_rejects_duplicate_categories() -> None:
    bounds = torch.tensor([[0.0], [2.0]])
    try:
        optimize_acqf_mixed_ga(
            _SumAcquisition(),
            bounds,
            q=1,
            categorical_values={0: [0.0, 1.0, 1.0]},
            population_size=8,
            generations=1,
        )
    except ValueError as exc:
        assert "duplicates" in str(exc)
    else:
        raise AssertionError("Duplicate categories must be rejected.")
