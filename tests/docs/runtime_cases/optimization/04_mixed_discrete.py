"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim.backends.botorch import optimize_acqf_mixed_botorch

base = torch.linspace(0, 1, 6, dtype=torch.double)
low = torch.stack((base, torch.zeros_like(base)), dim=-1)
high = torch.stack((base, torch.ones_like(base)), dim=-1)
train_X = torch.cat((low, high), dim=0)
train_Y = torch.sin(4 * train_X[:, :1]) + 0.1 * train_X[:, 1:]
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = PosteriorMean(model=model)
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
candidate, value = optimize_acqf_mixed_botorch(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    num_restarts=2,
    fixed_features_list=[{1: 0.0}, {1: 1.0}],
    raw_samples=16,
)
assert candidate.shape == (1, 2)
assert candidate[0, 1].item() in (0.0, 1.0)
assert torch.isfinite(value).all()
