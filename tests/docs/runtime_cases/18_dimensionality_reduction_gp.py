"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import PCAGP

X = torch.rand(12, 5, dtype=torch.double)
Y = (X[:, :1] - X[:, 1:2]).square()
model = PCAGP(train_X=X, train_Y=Y, n_components=2)
posterior = model.posterior(X[:2])
print(posterior.mean.shape)
