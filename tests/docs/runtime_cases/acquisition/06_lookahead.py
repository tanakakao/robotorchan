"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

from robotorchan.models import SingleTaskGP

train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
train_y = torch.sin(train_x * 5.0)
model = SingleTaskGP(train_x, train_y)
acqf = qKnowledgeGradient(model=model, num_fantasies=16)
augmented_q = acqf.get_augmented_q_batch_size(1)
assert augmented_q == 17
augmented_candidates = torch.linspace(0.1, 0.9, augmented_q, dtype=torch.double).reshape(
    1, augmented_q, 1
)
experimental_candidates = acqf.extract_candidates(augmented_candidates)
assert experimental_candidates.shape == (1, 1, 1)
assert torch.equal(experimental_candidates, augmented_candidates[..., :1, :])
