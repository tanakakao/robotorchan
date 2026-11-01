"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)

from robotorchan.models import SingleTaskGP

train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
train_y = torch.sin(train_x * 5.0)
model = SingleTaskGP(train_x, train_y)
candidate_set = torch.linspace(0.0, 1.0, 32, dtype=torch.double).unsqueeze(-1)
x = torch.tensor([[[0.4]]], dtype=torch.double)
mes = qMaxValueEntropy(model=model, candidate_set=candidate_set, num_mv_samples=4)
gibbon = qLowerBoundMaxValueEntropy(
    model=model,
    candidate_set=candidate_set,
    num_mv_samples=4,
)
mes_value = mes(x)
gibbon_value = gibbon(x)
