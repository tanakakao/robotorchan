"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import JointEncoderGP

train_X = torch.rand(16, 4, dtype=torch.double)
train_Y = (train_X[:, :1] - 0.4).square()
model = JointEncoderGP(
    train_X=train_X,
    train_Y=train_Y,
    latent_dim=2,
    hidden_dims=(16, 8),
)
print(model.latent_dim)
