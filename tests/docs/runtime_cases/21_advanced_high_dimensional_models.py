"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import AdditiveMapSaasSingleTaskGP

X = torch.rand(12, 6, dtype=torch.double)
Y = (X[:, :1] - 0.5).square()
model = AdditiveMapSaasSingleTaskGP(
    train_X=X,
    train_Y=Y,
    num_taus=4,
)
print(type(model).__name__)
