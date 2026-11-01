"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.analytic import UpperConfidenceBound

from robotorchan.models import SingleTaskGP
from robotorchan.optim.backends.sampling import (
    sample_candidate_batches,
    select_best_sampled_batch,
)

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = UpperConfidenceBound(model=model, beta=0.5)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
samples = sample_candidate_batches(bounds, num_samples=32, q=1, seed=7)
candidate, score = select_best_sampled_batch(acqf, samples)
assert samples.shape == (32, 1, 1)
assert candidate.shape == (1, 1)
assert torch.isfinite(score).all()
assert torch.all(candidate >= bounds[0])
assert torch.all(candidate <= bounds[1])
