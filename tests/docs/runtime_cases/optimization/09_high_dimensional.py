"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.analytic import PosteriorMean

from robotorchan.models import SingleTaskGP
from robotorchan.optim import HeSBOStrategy

train_X = torch.rand(12, 6, dtype=torch.double)
train_Y = -((train_X[:, :2] - 0.4) ** 2).sum(dim=-1, keepdim=True)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = PosteriorMean(model=model)
bounds = torch.stack((torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)))
strategy = HeSBOStrategy(bounds, embedding_dim=2, seed=7, num_restarts=2, raw_samples=16)
result = strategy.optimize(acqf)
original = result.candidates
embedded = result.metadata["embedded_candidates"]
assert original.shape == (1, 6)
assert embedded.shape == (1, 2)
assert strategy.embedding.shape == (6, 2)
assert torch.all(original >= bounds[0])
assert torch.all(original <= bounds[1])
torch.testing.assert_close(strategy.project(embedded), original)
