"""Runtime contract for native q-batch acquisitions and candidate optimization."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.models import SingleTaskGP


def test_native_qlogei_constraints_sampler_and_optimizer_end_to_end() -> None:
    """Native objective constraints are distinct from candidate-space constraints."""
    torch.manual_seed(13)
    X = torch.linspace(0.05, 0.95, 9, dtype=torch.double).unsqueeze(-1)
    Y = (X - 0.4).square()
    model = SingleTaskGP(X, Y)
    model.eval()

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=13)

    def outcome_constraint(samples: torch.Tensor) -> torch.Tensor:
        return samples[..., 0] - 0.6

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=0.0,
        sampler=sampler,
        constraints=[outcome_constraint],
    )
    assert check_acquisition_optimizer_compatibility(
        "qLogExpectedImprovement", CandidateOptimizer.CONTINUOUS, q=2
    ).compatible
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidate, value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 5},
    )
    assert candidate.shape == (2, 1)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
    assert ((candidate >= bounds[0]) & (candidate <= bounds[1])).all()
