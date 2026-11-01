"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.analytic import UpperConfidenceBound

from robotorchan.models import SingleTaskGP
from robotorchan.optim import optimize_acqf
from robotorchan.optim.constraints.contracts import CandidateConstraints

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = UpperConfidenceBound(model=model, beta=0.5)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
constraints = CandidateConstraints(
    inequality_constraints=(
        (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1.0], dtype=torch.double),
            0.3,
        ),
    ),
)
candidate, value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    optimizer="botorch",
    constraints=constraints,
    num_restarts=2,
    raw_samples=16,
)
assert candidate.shape == (1, 1)
assert candidate[0, 0].item() >= 0.3 - 1e-6
assert candidate[0, 0].item() <= 1.0 + 1e-6
assert torch.isfinite(value).all()
