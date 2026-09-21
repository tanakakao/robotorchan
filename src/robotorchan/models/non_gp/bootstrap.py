"""Shared infrastructure for bootstrap ensemble surrogates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from torch import Tensor, nn

from robotorchan.models.base import NonGPModelMixin
from robotorchan.models.non_gp.posterior import make_ensemble_posterior


class BootstrapEnsembleSurrogate(NonGPModelMixin, Model, nn.Module, ABC):
    """Base class for empirical posteriors built from complete bootstrap models."""

    model_name = "BootstrapEnsembleSurrogate"

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        n_members: int = 16,
        random_state: int | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        super().__init__()
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError(f"{self.model_name} supports one output only.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of observations.")
        if n_members < 2:
            raise ValueError("n_members must be at least 2.")

        self._store_supervised_training_data(train_X, train_Y)
        self.n_members = n_members
        self.random_state = random_state
        self.estimator_kwargs = dict(estimator_kwargs)
        self._members: list[Any] = []
        self._is_fitted = False

    @abstractmethod
    def _make_estimator(self, member_index: int) -> Any:
        """Construct one complete estimator."""

    @property
    def num_outputs(self) -> int:
        """Number of modeled outputs."""
        return 1

    @property
    def is_fitted(self) -> bool:
        """Whether all bootstrap members have been fitted."""
        return self._is_fitted

    def fit(self) -> None:
        """Fit complete estimators on independent bootstrap resamples."""
        X = self.raw_train_X.detach().cpu().numpy()
        y = self.raw_train_Y.squeeze(-1).detach().cpu().numpy()
        generator = torch.Generator().manual_seed(self.random_state or 0)
        self._members = []
        for member_index in range(self.n_members):
            indices = torch.randint(len(X), (len(X),), generator=generator).numpy()
            member = self._make_estimator(member_index)
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
        """Return predictions from complete bootstrap members."""
        del kwargs
        if not self._is_fitted:
            raise RuntimeError("Call fit() before posterior().")
        if output_indices not in (None, [0]):
            raise ValueError(f"{self.model_name} has only output index 0.")
        if observation_noise is not False:
            raise NotImplementedError(f"{self.model_name} does not model observation noise.")

        original_shape = X.shape[:-1]
        flat_X = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        predictions = [member.predict(flat_X) for member in self._members]
        values = torch.as_tensor(predictions, dtype=X.dtype, device=X.device)
        values = values.reshape(self.n_members, *original_shape, 1)
        values = values.movedim(0, len(original_shape) - 1)
        posterior = make_ensemble_posterior(values)
        return posterior_transform(posterior) if posterior_transform is not None else posterior
