"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import ContaminatedSingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(6 * train_X)
model = ContaminatedSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    num_inducing=4,
    num_likelihood_samples=4,
)
loss = model.training_loss()
posterior = model.posterior(train_X[:2])
diagnostic = model.contamination_diagnostic(train_X[:2], train_Y[:2])
assert loss.ndim == 0
assert posterior.mean.shape[-2:] == (2, 1)
assert diagnostic.shape == (2, 1)
