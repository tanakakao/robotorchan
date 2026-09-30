"""Cross-cutting optimizer behavior tests."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends import optimize_acqf_de, optimize_acqf_ga, optimize_acqf_pso
from robotorchan.optim.backend_support.operations import apply_fixed_features, optimize_acqf_sequential


class _Quadratic(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        target = torch.tensor([0.8, 0.2], dtype=X.dtype, device=X.device)
        return -((X - target) ** 2).sum(dim=(-1, -2))


class _PendingAware(_Quadratic):
    def __init__(self) -> None:
        super().__init__()
        self.X_pending = None

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = X_pending


def test_apply_fixed_features_supports_negative_indices() -> None:
    X = torch.zeros(3, 2, dtype=torch.double)
    result = apply_fixed_features(X, {-1: 0.4})
    assert torch.all(result[:, 1] == 0.4)


def test_derivative_free_backends_honor_fixed_features() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    cases = [
        (optimize_acqf_de, {"options": {"maxiter": 8, "popsize": 5}, "seed": 2}),
        (optimize_acqf_ga, {"population_size": 32, "generations": 10, "seed": 2}),
        (optimize_acqf_pso, {"swarm_size": 32, "iterations": 10, "seed": 2}),
    ]
    for optimizer, kwargs in cases:
        candidate, value = optimizer(_Quadratic(), bounds, 1, fixed_features={1: 0.6}, **kwargs)
        assert candidate.shape == (1, 2)
        assert float(candidate[0, 1]) == 0.6
        expected = _Quadratic()(candidate.unsqueeze(0)).reshape(())
        assert torch.equal(value, expected)


def test_sequential_helper_updates_and_restores_pending_points() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    acq = _PendingAware()
    original = torch.tensor([[0.1, 0.1]], dtype=torch.double)
    acq.set_X_pending(original)

    candidates, values = optimize_acqf_sequential(
        optimize_acqf_pso,
        acq,
        bounds,
        q=3,
        optimizer_kwargs={"swarm_size": 16, "iterations": 5, "seed": 3},
    )

    assert candidates.shape == (3, 2)
    assert values.shape == (3,)
    assert torch.equal(acq.X_pending, original)


def test_sequential_helper_restores_pending_when_optimizer_fails() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    acq = _PendingAware()
    original = torch.tensor([[0.1, 0.1]], dtype=torch.double)
    acq.set_X_pending(original)
    calls = 0

    def failing_optimizer(
        acq_function: AcquisitionFunction,
        optimizer_bounds: Tensor,
        q: int,
    ) -> tuple[Tensor, Tensor]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("optimizer failed")
        candidate = torch.tensor([[0.4, 0.6]], dtype=optimizer_bounds.dtype)
        return candidate, acq_function(candidate.unsqueeze(0)).reshape(())

    with pytest.raises(RuntimeError, match="optimizer failed"):
        optimize_acqf_sequential(failing_optimizer, acq, bounds, q=3)

    assert calls == 2
    assert torch.equal(acq.X_pending, original)
