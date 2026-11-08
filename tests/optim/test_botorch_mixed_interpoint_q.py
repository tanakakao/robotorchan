"""Inter-point q constraints must not reach sequential BoTorch mixed optimization."""

import pytest
import torch

from robotorchan.optim.backends.botorch import optimize_acqf_mixed_botorch
from robotorchan.optim.constraints import CandidateConstraints


def test_botorch_mixed_rejects_interpoint_linear_q_constraint() -> None:
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    constraints = CandidateConstraints(
        inequality_constraints=(
            (
                torch.tensor([[0, 0], [1, 0]]),
                torch.tensor([1.0, -1.0], dtype=torch.double),
                0.0,
            ),
        )
    )
    with pytest.raises(ValueError, match="does not support inter-point linear"):
        optimize_acqf_mixed_botorch(
            acq_function=None,
            bounds=bounds,
            q=2,
            num_restarts=1,
            fixed_features_list=[{1: 0.0}, {1: 1.0}],
            raw_samples=4,
            constraints=constraints,
        )
