"""NGBoost probabilistic regression surrogate."""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor, nn

from robotorchan.models.base import NonGPModelMixin
from robotorchan.models.non_gp.distribution_posterior import GaussianDistributionPosterior

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
except ImportError:  # pragma: no cover
    NGBRegressor = None
    Normal = None


class NGBoostSurrogate(NonGPModelMixin, Model, nn.Module):
    """Single-output Gaussian NGBoost surrogate with a sampleable posterior."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        random_state: int | None = None,
        **ngboost_kwargs: Any,
    ) -> None:
        super().__init__()
        if NGBRegressor is None or Normal is None:
            raise ImportError(
                "NGBoostSurrogate requires ngboost. Install the robotorchan ngboost extra."
            )
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("NGBoostSurrogate supports one output only.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of observations.")
        if not torch.is_floating_point(train_X) or not torch.is_floating_point(train_Y):
            raise ValueError("NGBoostSurrogate requires floating-point training tensors.")
        self._store_supervised_training_data(train_X, train_Y)
        self._estimator = NGBRegressor(
            Dist=Normal,
            random_state=random_state,
            **ngboost_kwargs,
        )
        self._is_fitted = False

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self) -> None:
        """Fit NGBoost using constructor-level raw training data."""
        X = self.raw_train_X.detach().cpu().numpy()
        y = self.raw_train_Y.squeeze(-1).detach().cpu().numpy()
        self._estimator.fit(X, y)
        self._is_fitted = True

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Posterior:
        """Return NGBoost's Gaussian predictive distribution as a BoTorch posterior."""
        del kwargs
        if not self._is_fitted:
            raise RuntimeError("Call fit() before posterior().")
        if output_indices not in (None, [0]):
            raise ValueError("NGBoostSurrogate has only output index 0.")
        if observation_noise is not False:
            raise NotImplementedError(
                "NGBoostSurrogate already represents its configured predictive distribution."
            )
        original_shape = X.shape[:-1]
        flat_X = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        distribution = self._estimator.pred_dist(flat_X)
        params = distribution.params
        mean = torch.as_tensor(params["loc"], dtype=X.dtype, device=X.device)
        scale = torch.as_tensor(params["scale"], dtype=X.dtype, device=X.device)
        mean = mean.reshape(*original_shape, 1)
        variance = scale.square().reshape(*original_shape, 1)
        posterior = GaussianDistributionPosterior(mean=mean, variance=variance)
        return posterior_transform(posterior) if posterior_transform is not None else posterior
