"""Deep Gaussian process models for binary classification."""

from __future__ import annotations

import gpytorch
import torch
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from torch import Tensor
from torch.distributions import Bernoulli

from robotorchan.models.classification.binary.validation import validate_binary_labels
from robotorchan.models.expressive.deep_gp import SingleTaskDeepGP


class BinarySingleTaskDeepGPClassifier(SingleTaskDeepGP):
    """Binary DeepGP classifier with a Bernoulli observation model."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        **kwargs: object,
    ) -> None:
        """Initialize a variational DeepGP from binary labels."""
        validate_binary_labels(train_Y)
        raw_train_Y = train_Y.detach().clone()
        model_train_Y = train_Y.unsqueeze(-1) if train_Y.ndim == 1 else train_Y
        model_train_Y = model_train_Y.to(dtype=train_X.dtype, device=train_X.device)
        super().__init__(train_X=train_X, train_Y=model_train_Y, **kwargs)
        self.likelihood = BernoulliLikelihood().to(train_X)
        self._store_raw_tensor("train_Y", raw_train_Y)

    def make_mll(self, num_data: int | None = None) -> DeepApproximateMLL:
        """Construct the Bernoulli DeepGP variational ELBO."""
        if num_data is None:
            num_data = self.raw_train_X.shape[-2]
        if num_data <= 0:
            raise ValueError("num_data must be positive.")
        return DeepApproximateMLL(
            VariationalELBO(
                likelihood=self.likelihood,
                model=self,
                num_data=num_data,
            )
        )

    def predictive_distribution(
        self,
        X: Tensor,
        *,
        num_samples: int | None = None,
    ) -> Bernoulli:
        """Return Bernoulli predictions integrated over DeepGP latent samples."""
        resolved_samples = self.posterior_samples if num_samples is None else int(num_samples)
        if resolved_samples < 1:
            raise ValueError("num_samples must be positive.")
        posterior = self.posterior(X, num_samples=resolved_samples)
        samples = posterior.rsample(torch.Size([resolved_samples]))
        positive = self.likelihood.forward(samples.squeeze(-1)).probs.mean(dim=0)
        return Bernoulli(probs=positive)

    def predict_proba(self, X: Tensor, *, num_samples: int | None = None) -> Tensor:
        """Return ``[P(y=0), P(y=1)]`` for each input."""
        positive = self.predictive_distribution(X, num_samples=num_samples).probs
        return torch.stack((1.0 - positive, positive), dim=-1)

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        num_samples: int | None = None,
    ) -> Tensor:
        """Return binary labels from posterior predictive probabilities."""
        if not isinstance(threshold, int | float):
            raise TypeError("threshold must be a real number.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        positive = self.predict_proba(X, num_samples=num_samples)[..., 1]
        return (positive >= threshold).to(dtype=torch.long)

    def predictive_variance(self, X: Tensor, *, num_samples: int | None = None) -> Tensor:
        """Return Bernoulli variance for both class probabilities."""
        probabilities = self.predict_proba(X, num_samples=num_samples)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, *, num_samples: int | None = None) -> Tensor:
        """Return predictive class entropy."""
        probabilities = self.predict_proba(X, num_samples=num_samples)
        terms = torch.special.xlogy(probabilities, probabilities)
        return -terms.sum(dim=-1)

    def training_loss(
        self,
        X: Tensor | None = None,
        Y: Tensor | None = None,
        *,
        num_data: int | None = None,
        num_likelihood_samples: int = 8,
    ) -> Tensor:
        """Return the negative Bernoulli DeepGP ELBO."""
        X = self.raw_train_X if X is None else X
        Y = self.raw_train_Y if Y is None else Y
        validate_binary_labels(Y)
        if Y.ndim == 2 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        if Y.ndim != 1:
            raise ValueError("Y must have shape n or n x 1.")
        if X.shape[-2] != Y.shape[0]:
            raise ValueError("X and Y must contain the same number of rows.")
        if num_likelihood_samples <= 0:
            raise ValueError("num_likelihood_samples must be positive.")
        target = Y.to(dtype=X.dtype, device=X.device)
        mll = self.make_mll(num_data=num_data)
        with gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
            return -mll(self(X), target)
