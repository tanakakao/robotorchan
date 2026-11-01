"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP

train_X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(2 * torch.pi * train_X)
train_Yvar = torch.full_like(train_Y, 0.01)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
model.eval()
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=7)
acqf = qLogNoisyExpectedImprovement(
    model=model,
    X_baseline=train_X,
    sampler=sampler,
    prune_baseline=False,
)
candidates = torch.tensor(
    [[[0.2], [0.7]], [[0.3], [0.8]]],
    dtype=torch.double,
)
scores = acqf(candidates)
assert scores.shape == (2,)
assert torch.isfinite(scores).all()
