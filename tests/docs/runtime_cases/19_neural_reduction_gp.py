"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import JointEncoderGP

X = torch.rand(16, 6, dtype=torch.double)
Y = (X[:, :1] - 0.3).square()
model = JointEncoderGP(
    train_X=X,
    train_Y=Y,
    latent_dim=2,
    hidden_dims=(16, 8),
)
print(model.latent_dim)
