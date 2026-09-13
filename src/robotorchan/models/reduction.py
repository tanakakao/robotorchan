"""Common reduction interfaces for high-dimensional GP wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt
from typing import Self

import torch
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

    Fitted metadata is stored in a persistent buffer so ``state_dict`` round
    trips preserve the reducer lifecycle as well as learned projection tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "_fit_metadata",
            torch.tensor([0, -1, -1], dtype=torch.long),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the reducer has been fitted."""
        return bool(self._fit_metadata[0].item())

    @property
    def input_dim(self) -> int:
        """Feature dimension seen during fitting."""
        self._check_fitted()
        return int(self._fit_metadata[1].item())

    @property
    def output_dim(self) -> int:
        """Reduced feature dimension produced by ``transform``."""
        self._check_fitted()
        return int(self._fit_metadata[2].item())

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

        self._fit_metadata.copy_(
            torch.tensor(
                [1, X.shape[-1], output_dim],
                dtype=self._fit_metadata.dtype,
                device=self._fit_metadata.device,
            )
        )
        return self

    def transform(self, X: Tensor) -> Tensor:
        """Transform the final feature dimension while preserving leading dims."""
        self._check_fitted()
        if X.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final dimension {self.input_dim}, got {X.shape[-1]}.")

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
        if not self.is_fitted:
            raise ReducerNotFittedError(f"{type(self).__name__} has not been fitted.")

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Materialize learned buffers before loading a reducer state dict.

        Reducer subclasses register learned tensors as ``None`` before fitting.
        PyTorch omits such buffers from an unfitted module's state structure, so
        they must be materialized from checkpoint shapes before the standard
        loader can copy the saved values.
        """
        for name, value in self._buffers.items():
            key = f"{prefix}{name}"
            if value is None and key in state_dict:
                self._buffers[name] = torch.empty_like(state_dict[key])

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    @abstractmethod
    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        """Fit from ``[n, d]`` data and return the reduced dimension."""

    @abstractmethod
    def _transform_2d(self, X: Tensor) -> Tensor:
        """Transform a two-dimensional tensor."""


class InputReducer(TensorReducer, ABC):
    """Reducer applied to candidate and training inputs before GP evaluation."""


class PCAInputReducer(InputReducer):
    """Principal-component projection for high-dimensional continuous inputs.

    The projection is fitted with ``torch.linalg.svd`` and stored as module
    buffers so it follows model device / dtype movement and serialization.
    Candidate tensors are always projected with the basis learned at ``fit``.

    Args:
        n_components: Number of principal components to retain.
        center: Whether to subtract the training-input mean before projection.
    """

    def __init__(self, n_components: int, *, center: bool = True) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)
        self.center = center
        self.register_buffer("mean", None)
        self.register_buffer("components", None)
        self.register_buffer("explained_variance", None)
        self.register_buffer("explained_variance_ratio", None)

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        del Y
        max_components = min(X.shape)
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds the PCA limit {max_components}."
            )

        mean = X.mean(dim=0) if self.center else torch.zeros_like(X[0])
        centered = X - mean
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)

        components = vh[: self.n_components].transpose(-2, -1).contiguous()
        denominator = max(X.shape[0] - 1, 1)
        variances = singular_values.square() / denominator
        total_variance = variances.sum()
        retained_variance = variances[: self.n_components]
        if total_variance > 0:
            ratio = retained_variance / total_variance
        else:
            ratio = torch.zeros_like(retained_variance)

        self.mean = mean
        self.components = components
        self.explained_variance = retained_variance
        self.explained_variance_ratio = ratio
        return self.n_components

    def _transform_2d(self, X: Tensor) -> Tensor:
        assert self.mean is not None
        assert self.components is not None
        centered = X - self.mean if self.center else X
        return centered @ self.components


class RandomProjectionInputReducer(InputReducer):
    """Gaussian random projection for high-dimensional continuous inputs.

    A single projection matrix is sampled during ``fit`` and then frozen. This
    is useful when a cheap, data-independent projection is preferred over PCA.

    Args:
        n_components: Number of projected dimensions.
        random_state: Seed used to generate the frozen projection matrix.
    """

    def __init__(self, n_components: int, *, random_state: int = 0) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.register_buffer("projection", None)

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        del Y
        if self.n_components > X.shape[-1]:
            raise ValueError(
                f"n_components={self.n_components} exceeds input dimension {X.shape[-1]}."
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_state)
        projection = torch.randn(
            X.shape[-1],
            self.n_components,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        ) / sqrt(self.n_components)
        self.projection = projection.to(device=X.device, dtype=X.dtype)
        return self.n_components

    def _transform_2d(self, X: Tensor) -> Tensor:
        assert self.projection is not None
        return X @ self.projection


class PLSInputReducer(InputReducer):
    """Supervised partial least-squares projection for continuous inputs.

    Components are extracted sequentially from the cross-covariance between
    residual ``X`` and ``Y``. The fitted rotation is frozen after ``fit`` and is
    applied to all later candidates, matching the reducer lifecycle required by
    sequential Bayesian optimization.

    Args:
        n_components: Number of latent PLS components.
        center: Whether to center ``X`` and ``Y`` before fitting.
        eps: Numerical threshold used when a component has negligible variance.
    """

    def __init__(
        self,
        n_components: int,
        *,
        center: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.n_components = int(n_components)
        self.center = center
        self.eps = float(eps)
        self.register_buffer("x_mean", None)
        self.register_buffer("y_mean", None)
        self.register_buffer("weights", None)
        self.register_buffer("loadings", None)
        self.register_buffer("rotation", None)

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        if Y is None:
            raise ValueError("PLSInputReducer requires paired target data Y.")
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)
        if Y.ndim != 2:
            raise ValueError("PLSInputReducer expects Y with shape [n, m].")

        max_components = min(X.shape[0] - 1, X.shape[-1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds the PLS limit {max_components}."
            )

        x_mean = X.mean(dim=0) if self.center else torch.zeros_like(X[0])
        y_mean = Y.mean(dim=0) if self.center else torch.zeros_like(Y[0])
        X_residual = X - x_mean
        Y_residual = Y - y_mean

        weights: list[Tensor] = []
        loadings: list[Tensor] = []
        for _ in range(self.n_components):
            cross_covariance = X_residual.transpose(-2, -1) @ Y_residual
            left_vectors, singular_values, _ = torch.linalg.svd(
                cross_covariance,
                full_matrices=False,
            )
            if singular_values.numel() == 0 or singular_values[0] <= self.eps:
                raise ValueError("PLS could not extract the requested number of components.")

            weight = left_vectors[:, 0]
            score = X_residual @ weight
            score_norm = score.square().sum()
            if score_norm <= self.eps:
                raise ValueError("PLS encountered a degenerate latent score.")

            x_loading = (X_residual.transpose(-2, -1) @ score) / score_norm
            y_loading = (Y_residual.transpose(-2, -1) @ score) / score_norm

            X_residual = X_residual - score.unsqueeze(-1) * x_loading.unsqueeze(0)
            Y_residual = Y_residual - score.unsqueeze(-1) * y_loading.unsqueeze(0)
            weights.append(weight)
            loadings.append(x_loading)

        weight_matrix = torch.stack(weights, dim=-1)
        loading_matrix = torch.stack(loadings, dim=-1)
        inner = loading_matrix.transpose(-2, -1) @ weight_matrix
        rotation = weight_matrix @ torch.linalg.pinv(inner)

        self.x_mean = x_mean
        self.y_mean = y_mean
        self.weights = weight_matrix
        self.loadings = loading_matrix
        self.rotation = rotation
        return self.n_components

    def _transform_2d(self, X: Tensor) -> Tensor:
        assert self.x_mean is not None
        assert self.rotation is not None
        centered = X - self.x_mean if self.center else X
        return centered @ self.rotation


class OutputReducer(TensorReducer, ABC):
    """Reducer applied to training outcomes and restored after GP prediction."""

    def inverse_transform(self, Y: Tensor) -> Tensor:
        """Restore reduced values to the original outcome space."""
        self._check_fitted()
        if Y.shape[-1] != self.output_dim:
            raise ValueError(f"Expected final dimension {self.output_dim}, got {Y.shape[-1]}.")

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
