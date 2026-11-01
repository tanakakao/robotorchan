"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.analytic import UpperConfidenceBound

from robotorchan.models import SingleTaskGP
from robotorchan.optim import optimize_acqf

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = UpperConfidenceBound(model=model, beta=0.5)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
initial_conditions = torch.tensor([[[0.2]], [[0.8]]], dtype=torch.double)
candidate, value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    optimizer="botorch",
    num_restarts=2,
    raw_samples=None,
    batch_initial_conditions=initial_conditions,
)
assert candidate.shape == (1, 1)
assert torch.isfinite(value).all()
assert torch.all(candidate >= bounds[0])
assert torch.all(candidate <= bounds[1])
