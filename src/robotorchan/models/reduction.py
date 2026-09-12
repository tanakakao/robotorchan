"""Common reduction interfaces for high-dimensional GP wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from botorch.posteriors import Posterior
from torch import Tensor
from torch.nn import Module


class ReducerNotFittedError(RuntimeError):
    """Raised when a reducer is used before fitting."""


class TensorReducer(Module, ABC):
    """Base class for tensor reducers used by reduced GP models.

    Subclasses learn their state from two-dimensional training tensors while
    ``transform`` must accept arbitrary leading batch dimensions and only alter
    the final feature dimension. This contract keeps reduced models compatible
    with BoTorch q-batches, fantasies, and input perturbation workflows.
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_fitted = False
        self._input_dim: int | None = None
        self._output_dim: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the reducer has been fitted."""
        return self._is_fitted

    @property
    def input_dim(self) -> int:
        """Feature dimension seen during fitting."""
        self._check_fitted()
        assert self._input_dim is not None
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Reduced feature dimension produced by ``transform``."""
        self._check_fitted()
        assert self._output_dim is not None
        return self._output_dim

    def fit(self, X: Tensor, Y: Tensor | None = None) -> Self:
        """Fit the reducer from a two-dimensional training tensor.

        ``Y`` is optional because supervised reducers such as PLS may require
        the paired target tensor while unsupervised reducers such as PCA do not.
        """
        if X.ndim != 2:
            raise ValueError("Reducer.fit() expects X with shape [n, d].")
        if Y is not None and Y.shape[0] != X.shape[0]:
            raise ValueError("X and Y must have the same number of observations.")

        output_dim = self._fit_2d(X, Y)
        if output_dim <= 0:
            raise ValueError("A reducer must produce at least one output dimension.")

        self._input_dim = X.shape[-1]
        self._output_dim = output_dim
        self._is_fitted = True
        return self

    def transform(self, X: Tensor) -> Tensor:
        """Transform the final feature dimension while preserving leading dims."""
        self._check_fitted()
        if X.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected final dimension {self.input_dim}, got {X.shape[-1]}."
            )

        original_shape = X.shape
        X_2d = X.reshape(-1, original_shape[-1])
        transformed = self._transform_2d(X_2d)
        expected_shape = (X_2d.shape[0], self.output_dim)
        if transformed.shape != expected_shape:
            raise RuntimeError(
                "Reducer returned an invalid shape: "
                f"expected {expected_shape}, got {tuple(transformed.shape)}."
            )
        return transformed.reshape(*original_shape[:-1], self.output_dim)

    def fit_transform(self, X: Tensor, Y: Tensor | None = None) -> Tensor:
        """Fit the reducer and transform ``X``."""
        return self.fit(X, Y).transform(X)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise ReducerNotFittedError(f"{type(self).__name__} has not been fitted.")

    @abstractmethod
    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        """Fit from ``[n, d]`` data and return the reduced dimension."""

    @abstractmethod
    def _transform_2d(self, X: Tensor) -> Tensor:
        """Transform a two-dimensional tensor."""


class InputReducer(TensorReducer, ABC):
    """Reducer applied to candidate and training inputs before GP evaluation."""


class OutputReducer(TensorReducer, ABC):
    """Reducer applied to training outcomes and restored after GP prediction."""

    def inverse_transform(self, Y: Tensor) -> Tensor:
        """Restore reduced values to the original outcome space."""
        self._check_fitted()
        if Y.shape[-1] != self.output_dim:
            raise ValueError(
                f"Expected final dimension {self.output_dim}, got {Y.shape[-1]}."
            )

        original_shape = Y.shape
        Y_2d = Y.reshape(-1, original_shape[-1])
        restored = self._inverse_transform_2d(Y_2d)
        expected_shape = (Y_2d.shape[0], self.input_dim)
        if restored.shape != expected_shape:
            raise RuntimeError(
                "Reducer returned an invalid inverse shape: "
                f"expected {expected_shape}, got {tuple(restored.shape)}."
            )
        return restored.reshape(*original_shape[:-1], self.input_dim)

    @abstractmethod
    def _inverse_transform_2d(self, Y: Tensor) -> Tensor:
        """Restore a two-dimensional reduced tensor."""

    @abstractmethod
    def restore_posterior(self, posterior: Posterior) -> Posterior:
        """Restore a latent posterior to the original outcome space.

        Implementations must restore uncertainty, not only the posterior mean.
        Linear reducers such as PCA should therefore propagate covariance into
        the original output space.
        """


class ReductionMixin:
    """Shared reducer plumbing for future reduced GP wrappers."""

    input_reducer: InputReducer | None
    output_reducer: OutputReducer | None

    def _set_reducers(
        self,
        input_reducer: InputReducer | None,
        output_reducer: OutputReducer | None,
    ) -> None:
        self.input_reducer = input_reducer
        self.output_reducer = output_reducer

    def _fit_transform_inputs(self, train_X: Tensor, train_Y: Tensor) -> Tensor:
        if self.input_reducer is None:
            return train_X
        return self.input_reducer.fit_transform(train_X, train_Y)

    def _fit_transform_outputs(self, train_X: Tensor, train_Y: Tensor) -> Tensor:
        if self.output_reducer is None:
            return train_Y
        return self.output_reducer.fit_transform(train_Y, train_X)

    def _transform_inputs(self, X: Tensor) -> Tensor:
        if self.input_reducer is None:
            return X
        return self.input_reducer.transform(X)

    def _restore_output_posterior(self, posterior: Posterior) -> Posterior:
        if self.output_reducer is None:
            return posterior
        return self.output_reducer.restore_posterior(posterior)
