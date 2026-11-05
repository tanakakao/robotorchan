"""Phase 9: registered joint-q acquisitions with real GP posterior and optimizers."""

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.optim.backends import optimize_acqf_botorch, optimize_acqf_sampling


@pytest.mark.parametrize("q", [2, 4])
@pytest.mark.parametrize(
    "acquisition_name",
    [
        "qLogExpectedImprovement",
        "qProbabilityOfImprovement",
        "qSimpleRegret",
        "qUpperConfidenceBound",
    ],
)
@pytest.mark.parametrize("optimizer_name", ["botorch", "sobol"])
def test_registered_joint_acquisition_optimizer_q_batch(
    acquisition_name: str,
    optimizer_name: str,
    q: int,
) -> None:
    train_x = torch.linspace(0.0, 1.0, 7, dtype=torch.double).unsqueeze(-1)
    train_y = 1.0 - (train_x - 0.7).square()
    model = SingleTaskGP(train_x, train_y)
    model.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]))
    acquisitions = {
        "qLogExpectedImprovement": lambda: qLogExpectedImprovement(
            model, best_f=train_y.max(), sampler=sampler
        ),
        "qProbabilityOfImprovement": lambda: qProbabilityOfImprovement(
            model, best_f=train_y.max(), sampler=sampler
        ),
        "qSimpleRegret": lambda: qSimpleRegret(model, sampler=sampler),
        "qUpperConfidenceBound": lambda: qUpperConfidenceBound(model, beta=0.1, sampler=sampler),
    }
    acquisition = acquisitions[acquisition_name]()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    torch.manual_seed(7)
    if optimizer_name == "botorch":
        candidates, value = optimize_acqf_botorch(
            acquisition, bounds, q=q, num_restarts=2, raw_samples=16
        )
    else:
        candidates, value = optimize_acqf_sampling(
            acquisition, bounds, q=q, num_samples=48, method="sobol", seed=7
        )

    assert candidates.shape == (q, 1)
    assert value.numel() == 1
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert torch.all(candidates >= bounds[0])
    assert torch.all(candidates <= bounds[1])
    with torch.no_grad():
        actual = acquisition(candidates.unsqueeze(0)).reshape(())
    torch.testing.assert_close(value.reshape(()), actual, atol=1e-5, rtol=1e-4)
