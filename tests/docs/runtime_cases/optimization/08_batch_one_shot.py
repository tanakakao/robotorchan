"""Runtime inputs retained when usage examples were removed from Guide."""

import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

from robotorchan.models import SingleTaskGP

train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
train_Y = torch.sin(5 * train_X)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = qKnowledgeGradient(model=model, num_fantasies=4)
q = 2
augmented_q = acqf.get_augmented_q_batch_size(q)
augmented = torch.linspace(0.1, 0.9, augmented_q, dtype=torch.double)
augmented = augmented.reshape(1, augmented_q, 1)
candidates = acqf.extract_candidates(augmented)
assert augmented_q == 6
assert augmented.shape == (1, 6, 1)
assert candidates.shape == (1, 2, 1)
assert torch.equal(candidates, augmented[:, :q, :])
