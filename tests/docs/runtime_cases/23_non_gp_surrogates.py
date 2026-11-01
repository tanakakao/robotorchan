"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import RandomForestSurrogate

train_X = torch.rand(20, 3, dtype=torch.double)
train_Y = train_X[:, :1].square()
model = RandomForestSurrogate(train_X=train_X, train_Y=train_Y)
model.fit()
posterior = model.posterior(train_X[:2])
print(posterior.mean.shape)
