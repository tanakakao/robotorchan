"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.optim import TuRBOStrategy

bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
center = torch.tensor([0.5, 0.5], dtype=torch.double)
strategy = TuRBOStrategy(bounds, center=center)
print(strategy.state.length)
