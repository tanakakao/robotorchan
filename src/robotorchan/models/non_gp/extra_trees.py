"""Extra-trees surrogate models for Bayesian optimization."""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor, nn

from robotorchan.models.base import NonGPModelMixin
from robotorchan.models.non_gp.posterior import make_ensemble_posterior

try:
    from sklearn.ensemble import ExtraTreesRegressor
except ImportError:  # pragma: no cover
    ExtraTreesRegressor = None


class ExtraTreesSurrogate(NonGPModelMixin, Model, nn.Module):
    """BoTorch-compatible extremely randomized trees regression surrogate."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        n_estimators: int = 100,
        random_state: int | None = None,
        **forest_kwargs: Any,
    ) -> None:
        super().__init__()
        if ExtraTreesRegressor is None:
            raise ImportError(
                "ExtraTreesSurrogate requires scikit-learn. Install the robotorchan tree extra."
            )
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("Phase 5 ExtraTreesSurrogate supports one output only.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of observations.")
        self._store_supervised_training_data(train_X, train_Y)
        self._forest = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **forest_kwargs,
        )
        self._is_fitted = False

    @property
    def num_outputs(self) -> int:
        """Number of modeled outputs."""
        return 1

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying ensemble has been fitted."""
        return self._is_fitted

    def fit(self) -> None:
        """Fit the extra-trees ensemble using constructor-level raw training data."""
        X = self.raw_train_X.detach().cpu().numpy()
        y = self.raw_train_Y.squeeze(-1).detach().cpu().numpy()
        self._forest.fit(X, y)
        self._is_fitted = True

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> Posterior:
        """Return empirical tree predictions as a BoTorch ensemble posterior."""
        del kwargs
        if not self._is_fitted:
            raise RuntimeError("Call fit() before posterior().")
        if output_indices not in (None, [0]):
            raise ValueError("ExtraTreesSurrogate has only output index 0.")
        if observation_noise is not False:
            raise NotImplementedError("ExtraTreesSurrogate does not model observation noise.")

        original_shape = X.shape[:-1]
        flat_X = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        tree_predictions = [tree.predict(flat_X) for tree in self._forest.estimators_]
        values = torch.as_tensor(tree_predictions, device=X.device, dtype=X.dtype)
        values = values.reshape(len(tree_predictions), *original_shape, 1)
        ensemble_dim = len(original_shape) - 1
        values = values.movedim(0, ensemble_dim)
        posterior = make_ensemble_posterior(values)
        return posterior_transform(posterior) if posterior_transform is not None else posterior
