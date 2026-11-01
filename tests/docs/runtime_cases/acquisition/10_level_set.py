"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.acquisition import Straddle
from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
candidates = torch.tensor([[[0.15]], [[0.45]], [[0.85]]], dtype=torch.double)
target = 0.35
beta = 1.7
scores = Straddle(model, target=target, beta=beta)(candidates)
assert scores.shape == (3,)
assert torch.isfinite(scores).all()
