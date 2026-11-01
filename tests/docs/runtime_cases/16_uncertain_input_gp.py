"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import UncertainInputSingleTaskGP

train_X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(5 * train_X)
train_X_std = torch.full_like(train_X, 0.02)
model = UncertainInputSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    train_X_std=train_X_std,
)
candidate_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
posterior = model.posterior(candidate_X)
assert posterior.mean.shape[-2:] == (2, 1)
