"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from robotorchan.models import ModelListGP, SingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
y1 = torch.sin(5 * train_X)
y2 = torch.cos(5 * train_X)
model = ModelListGP(SingleTaskGP(train_X, y1), SingleTaskGP(train_X, y2))
model.eval()
train_Y = torch.cat((y1, y2), dim=-1)
ref_point = train_Y.min(dim=0).values - 0.1
partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y)
acqf = qLogExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point.tolist(),
    partitioning=partitioning,
)
candidates = torch.tensor([[[0.4]]], dtype=torch.double)
score = acqf(candidates)
assert score.shape == (1,)
assert torch.isfinite(score).all()
