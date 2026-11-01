"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import NonstationarySingleTaskGP

X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
Y = torch.sin(6 * X)
model = NonstationarySingleTaskGP(train_X=X, train_Y=Y)
print(model.gibbs_kernel.local_lengthscale(X).shape)
