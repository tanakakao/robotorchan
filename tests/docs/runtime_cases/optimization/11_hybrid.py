"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import optimize_acqf

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = PosteriorMean(model=model)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
candidate, score = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    optimizer="hybrid",
    seed=7,
    optimizer_options={
        "global_optimizer": "de",
        "global_options": {"options": {"maxiter": 2, "popsize": 4}},
        "num_restarts": 1,
    },
)
assert candidate.shape == (1, 1)
assert torch.isfinite(candidate).all()
assert torch.isfinite(score).all()
assert torch.all(candidate >= bounds[0])
assert torch.all(candidate <= bounds[1])
