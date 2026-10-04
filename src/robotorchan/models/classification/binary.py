"""Binary variational GP classifier."""

from __future__ import annotations

import torch

from botorch.models.transforms.input import InputTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from gpytorch.variational import VariationalStrategy, _VariationalDistribution, _VariationalStrategy
from torch import Tensor
from torch.distributions import Bernoulli

from robotorchan.models.classification.base import BinaryClassificationMixin
from robotorchan.models.classification.validation import validate_binary_labels
from robotorchan.models.standard.variational import SingleTaskVariationalGP


class BinarySingleTaskGPClassifier(BinaryClassificationMixin, SingleTaskVariationalGP):
    """Binary GP classifier backed by BoTorch variational GP infrastructure."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        learn_inducing_points: bool = True,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a binary variational GP with Bernoulli likelihood."""
        validate_binary_labels(train_Y)
        model_train_Y = train_Y.unsqueeze(-1) if train_Y.ndim == train_X.ndim - 1 else train_Y
        super().__init__(
            train_X=train_X,
            train_Y=model_train_Y,
            likelihood=BernoulliLikelihood(),
            num_outputs=1,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            outcome_transform=None,
            input_transform=input_transform,
        )
        self._store_raw_tensor("train_Y", train_Y.detach().clone())

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Bernoulli:
        """Return Bernoulli predictions after integrating latent uncertainty."""
        latent = self.latent_posterior(X, **kwargs)
        return self.likelihood(latent.distribution)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return ``[P(y=0), P(y=1)]`` along the final dimension."""
        positive = self.predictive_distribution(X, **kwargs).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return binary class labels using the requested probability threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        return (self.predict_proba(X, **kwargs)[..., 1] >= threshold).long()
