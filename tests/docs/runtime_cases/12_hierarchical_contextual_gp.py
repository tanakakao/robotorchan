"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import MixedHierarchicalConditionalKernelGP

train_X = torch.tensor(
    [
        [0.0, 0.10, 0.0, 0.20],
        [1.0, 0.25, 0.0, 0.35],
        [0.0, 0.40, 1.0, 0.55],
        [1.0, 0.60, 1.0, 0.70],
        [0.0, 0.75, 0.0, 0.80],
        [1.0, 0.90, 1.0, 0.95],
    ],
    dtype=torch.double,
)
train_Y = 0.3 * train_X[:, :1] + train_X[:, 1:2] + 0.2 * train_X[:, 3:4]
dependencies = {2: {0: [1], 1: [3]}}
model = MixedHierarchicalConditionalKernelGP(
    train_X=train_X,
    train_Y=train_Y,
    hierarchical_dependencies=dependencies,
    cat_dims=[0],
    use_saas_prior=False,
    outcome_transform=None,
)
assert model.cat_dims == (0,)
assert model.raw_train_X.shape == train_X.shape
assert model.make_mll() is not None
