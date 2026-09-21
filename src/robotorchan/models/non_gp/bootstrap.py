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
        if train_Y.ndim != 2 or train_Y.shape[-1] < 1:
            raise ValueError("train_Y must have shape n x m with at least one output.")
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
        return self.raw_train_Y.shape[-1]

    @property
    def is_fitted(self) -> bool:
        """Whether all bootstrap members have been fitted."""
        return self._is_fitted

    def fit(self) -> None:
        """Fit complete estimators on independent bootstrap resamples."""
        X = self.raw_train_X.detach().cpu().numpy()
        y = self.raw_train_Y.detach().cpu().numpy()
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
        selected_outputs = (
            list(range(self.num_outputs)) if output_indices is None else output_indices
        )
        if not selected_outputs or any(
            index < 0 or index >= self.num_outputs for index in selected_outputs
        ):
            raise ValueError("output_indices contains an invalid output index.")
        if observation_noise is not False:
            raise NotImplementedError(f"{self.model_name} does not model observation noise.")

        original_shape = X.shape[:-1]
        flat_X = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        predictions = []
        for member in self._members:
            prediction = torch.as_tensor(member.predict(flat_X))
            if prediction.ndim == 1:
                prediction = prediction.unsqueeze(-1)
            predictions.append(prediction[..., selected_outputs])
        values = torch.stack(predictions).to(dtype=X.dtype, device=X.device)
        values = values.reshape(self.n_members, *original_shape, len(selected_outputs))
        values = values.movedim(0, len(original_shape) - 1)
        posterior = make_ensemble_posterior(values)
        return posterior_transform(posterior) if posterior_transform is not None else posterior
