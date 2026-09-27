"""Tests for hybrid global-to-local acquisition optimization."""

from unittest.mock import Mock, patch

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor

from robotorchan.optim.backends.hybrid import optimize_acqf_hybrid
from robotorchan.optim.constraints import CandidateConstraints


class _Quadratic(AcquisitionFunction):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, X: Tensor) -> Tensor:
        return -((X - 0.73) ** 2).sum(dim=(-2, -1))


def test_hybrid_improves_or_preserves_global_candidate() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    global_candidate = torch.tensor([[0.4]], dtype=torch.double)

    def global_optimizer(*args, **kwargs):
        return global_candidate, torch.tensor(-0.1, dtype=torch.double)

    candidate, value = optimize_acqf_hybrid(
        _Quadratic(),
        bounds,
        q=1,
        global_optimizer=global_optimizer,
        local_options={"maxiter": 50},
    )

    assert abs(float(candidate[0, 0]) - 0.73) < 1e-3
    assert float(value) >= float(_Quadratic()(global_candidate.unsqueeze(0)))


def test_hybrid_forwards_constraints_to_global_and_local_stages() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=((torch.tensor([0]), torch.tensor([-1.0]), -0.4),)
    )
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.3]], dtype=torch.double), torch.tensor(-0.1))
    )
    local_candidate = torch.tensor([[0.35]], dtype=torch.double)
    local_value = torch.tensor(-0.01, dtype=torch.double)

    with patch(
        "robotorchan.optim.backends.hybrid.botorch_optimize_acqf",
        return_value=(local_candidate, local_value),
    ) as local_optimizer:
        candidate, value = optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            constraints=constraints,
            seed=5,
        )

    assert torch.equal(candidate, local_candidate)
    assert torch.equal(value, local_value)
    assert global_optimizer.call_args.kwargs["constraints"] is constraints
    local_kwargs = local_optimizer.call_args.kwargs
    assert local_kwargs["inequality_constraints"] == list(constraints.inequality_constraints)
    assert torch.equal(local_kwargs["batch_initial_conditions"], torch.tensor([[[0.3]]]))


def test_hybrid_sets_batch_limit_for_nonlinear_constraints() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        nonlinear_inequality_constraints=((lambda x: 0.5 - x[..., 0], True),)
    )
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.3]], dtype=torch.double), torch.tensor(-0.1))
    )

    with patch(
        "robotorchan.optim.backends.hybrid.botorch_optimize_acqf",
        return_value=(torch.tensor([[0.3]]), torch.tensor(-0.1)),
    ) as local_optimizer:
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            constraints=constraints,
        )

    assert local_optimizer.call_args.kwargs["options"]["batch_limit"] == 1


def test_hybrid_rejects_duplicate_restarts_and_fixed_features() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    global_optimizer = Mock(
        return_value=(torch.tensor([[0.3]], dtype=torch.double), torch.tensor(-0.1))
    )

    with pytest.raises(ValueError, match="num_restarts > 1"):
        optimize_acqf_hybrid(
            _Quadratic(), bounds, q=1, global_optimizer=global_optimizer, num_restarts=2
        )
    with pytest.raises(NotImplementedError, match="fixed_features"):
        optimize_acqf_hybrid(
            _Quadratic(),
            bounds,
            q=1,
            global_optimizer=global_optimizer,
            fixed_features={0: 0.5},
        )


def test_hybrid_rejects_unknown_global_optimizer() -> None:
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="global_optimizer"):
        optimize_acqf_hybrid(_Quadratic(), bounds, q=1, global_optimizer="unknown")  # type: ignore[arg-type]
