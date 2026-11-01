"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.analytic import LogExpectedImprovement

from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(2 * torch.pi * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
candidates = torch.tensor([[[0.25]], [[0.75]]], dtype=torch.double)
best_f = train_Y.max()
acqf = LogExpectedImprovement(model=model, best_f=best_f)
scores = acqf(candidates)
assert scores.shape == (2,)
assert torch.isfinite(scores).all()
