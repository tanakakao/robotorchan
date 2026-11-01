"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import StudentTSingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(6 * train_X)
model = StudentTSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    num_inducing=4,
    df=4.0,
)
loss = model.training_loss()
posterior = model.posterior(train_X[:2])
assert loss.ndim == 0
assert posterior.mean.shape[-2:] == (2, 1)
