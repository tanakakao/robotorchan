"""Quantile gradient-boosting surrogate for Bayesian optimization."""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor, nn

from robotorchan.models.base import NonGPModelMixin
from robotorchan.models.non_gp.posterior import make_ensemble_posterior

try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:  # pragma: no cover
    GradientBoostingRegressor = None


class GradientBoostingSurrogate(NonGPModelMixin, Model, nn.Module):
    """Gradient boosting with an empirical bootstrap predictive ensemble.

    Each posterior member is a complete independently fitted boosting model.
    Individual boosting stages are never interpreted as posterior samples.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        n_members: int = 16,
        random_state: int | None = None,
        **boosting_kwargs: Any,
    ) -> None:
        super().__init__()
        if GradientBoostingRegressor is None:
            raise ImportError(
                "GradientBoostingSurrogate requires scikit-learn. "
                "Install the robotorchan tree extra."
            )
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("GradientBoostingSurrogate supports one output only.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of observations.")
        if n_members < 2:
            raise ValueError("n_members must be at least 2.")

        self._store_supervised_training_data(train_X, train_Y)
        self.n_members = n_members
        self.random_state = random_state
        self.boosting_kwargs = dict(boosting_kwargs)
        self._members: list[GradientBoostingRegressor] = []
        self._is_fitted = False

    @property
    def num_outputs(self) -> int:
        """Number of modeled outputs."""
        return 1

    @property
    def is_fitted(self) -> bool:
        """Whether the bootstrap ensemble has been fitted."""
        return self._is_fitted

    def fit(self) -> None:
        """Fit independent boosting models on bootstrap resamples."""
        X = self.raw_train_X.detach().cpu().numpy()
        y = self.raw_train_Y.squeeze(-1).detach().cpu().numpy()
        generator = torch.Generator().manual_seed(self.random_state or 0)
        self._members = []
        for member_index in range(self.n_members):
            indices = torch.randint(len(X), (len(X),), generator=generator).numpy()
            member = GradientBoostingRegressor(
                random_state=(
                    None if self.random_state is None else self.random_state + member_index
                ),
                **self.boosting_kwargs,
            )
            member.fit(X[indices], y[indices])
            self._members.append(member)
        self._is_fitted = True

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Posterior:
        """Return complete boosting-model predictions as an ensemble posterior."""
        del kwargs
        if not self._is_fitted:
            raise RuntimeError("Call fit() before posterior().")
        if output_indices not in (None, [0]):
            raise ValueError("GradientBoostingSurrogate has only output index 0.")
        if observation_noise is not False:
            raise NotImplementedError("GradientBoostingSurrogate does not model observation noise.")

        original_shape = X.shape[:-1]
        flat_X = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        predictions = [member.predict(flat_X) for member in self._members]
        values = torch.as_tensor(predictions, dtype=X.dtype, device=X.device)
        values = values.reshape(self.n_members, *original_shape, 1)
        values = values.movedim(0, len(original_shape) - 1)
        posterior = make_ensemble_posterior(values)
        return posterior_transform(posterior) if posterior_transform is not None else posterior
