"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.models.cost import AffineFidelityCostModel

cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
candidates = torch.tensor([[[0.3, 0.5]], [[0.3, 1.0]]], dtype=torch.double)
costs = cost_model(candidates)
assert torch.isfinite(costs).all()
assert torch.all(costs > 0)
assert costs.reshape(-1)[0] < costs.reshape(-1)[1]
