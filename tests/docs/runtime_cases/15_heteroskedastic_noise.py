"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models import JointHeteroskedasticSingleTaskGP

train_X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(5 * train_X)
model = JointHeteroskedasticSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    num_inducing=4,
    num_mc_samples=4,
)
loss = model.training_loss()
candidate_X = train_X[:2]
response = model.posterior(candidate_X)
log_noise = model.noise_posterior(candidate_X)
observation_variance = model.predicted_noise(candidate_X)
assert loss.ndim == 0
assert response.mean.shape[-2:] == (2, 1)
assert log_noise.mean.shape[-2:] == (2, 1)
assert observation_variance.shape[-2:] == (2, 1)
