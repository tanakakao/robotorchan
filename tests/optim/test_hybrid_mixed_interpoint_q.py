"""Mixed Hybrid must not silently discard inter-point q-batch constraints."""

import pytest
import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from robotorchan.optim.backends.hybrid import optimize_acqf_hybrid
from robotorchan.optim.constraints import CandidateConstraints
from robotorchan.optim.domains import MixedVariableSpace


class _SumAcquisition(AcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(-1, -2))


def test_mixed_hybrid_rejects_interpoint_linear_q_batch() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, categorical_values={1: (0.0, 1.0)})
    constraints = CandidateConstraints(
        equality_constraints=(
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([1.0, -1.0], dtype=torch.double),
                0.0,
            ),
        )
    )
    with pytest.raises(ValueError, match="cannot preserve inter-point linear"):
        optimize_acqf_hybrid(
            _SumAcquisition(),
            bounds,
            q=2,
            variable_space=space,
            mixed_fixed_features_list=[{1: 0.0}, {1: 1.0}],
            constraints=constraints,
        )
