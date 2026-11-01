"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.acquisition.composition import make_continuous_constrained_qei_acquisition
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard import SingleTaskGP
from robotorchan.semantics import (
    ContinuousConstraint,
    ProblemSemantics,
    RegressionObjective,
)

train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
train_y = torch.cat((torch.sin(train_x), torch.cos(train_x)), dim=-1)
model = HeterogeneousModel(SingleTaskGP(train_x, train_y))
semantics = ProblemSemantics(
    objectives=(RegressionObjective(0),),
    constraints=(ContinuousConstraint(1, threshold=0.8),),
)
acq = make_continuous_constrained_qei_acquisition(
    model,
    semantics,
    best_f=0.5,
    infeasible_cost=0.0,
)
value = acq(torch.tensor([[0.25]], dtype=torch.double))
assert value.shape == torch.Size([1])
assert torch.isfinite(value).all()
