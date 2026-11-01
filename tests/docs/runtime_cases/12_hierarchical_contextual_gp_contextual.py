"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import MixedLCEMGP

train_X = torch.tensor(
    [
        [0.10, 0.0, 0.0],
        [0.20, 1.0, 0.0],
        [0.40, 0.0, 1.0],
        [0.55, 1.0, 1.0],
        [0.75, 0.0, 0.0],
        [0.90, 1.0, 1.0],
    ],
    dtype=torch.double,
)
train_Y = train_X[:, :1] + 0.25 * train_X[:, 1:2] + 0.4 * train_X[:, 2:3]
model = MixedLCEMGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=-1,
    cat_dims=[1],
    outcome_transform=None,
)
assert model.cat_dims == (1,)
assert model.raw_task_feature == 2
assert model.make_mll() is not None
