"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import PairwiseGP

points = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
comparisons = torch.tensor([[1, 0], [1, 2]], dtype=torch.long)
model = PairwiseGP(datapoints=points, comparisons=comparisons)
print(type(model.make_mll()).__name__)
