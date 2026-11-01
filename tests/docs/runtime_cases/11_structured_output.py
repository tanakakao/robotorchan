"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import HigherOrderGP

torch.manual_seed(0)
X = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double)
Y = torch.rand(3, 2, 2, dtype=torch.double)
model = HigherOrderGP(train_X=X, train_Y=Y)
print(type(model).__name__)
