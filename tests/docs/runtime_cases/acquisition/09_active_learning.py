"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.acquisition import PosteriorStd, PosteriorVariance
from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(4 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
candidates = torch.tensor([[[0.15]], [[0.45]], [[0.85]]], dtype=torch.double)
variance_score = PosteriorVariance(model)(candidates)
std_score = PosteriorStd(model)(candidates)
assert variance_score.shape == (3,)
assert torch.isfinite(variance_score).all()
assert torch.isfinite(std_score).all()
