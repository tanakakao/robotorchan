"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.models.classification import BinarySingleTaskGPClassifier

train_X = torch.rand(20, 3, dtype=torch.double)
train_Y = torch.randint(0, 2, (20,), dtype=torch.double)
model = BinarySingleTaskGPClassifier(train_X, train_Y)

X = torch.rand(5, 3, dtype=torch.double)
latent_posterior = model.posterior(X)
class_probabilities = model.predict_proba(X)
predicted_labels = model.predict_class(X)
print(latent_posterior.mean.shape, class_probabilities.shape, predicted_labels.shape)
