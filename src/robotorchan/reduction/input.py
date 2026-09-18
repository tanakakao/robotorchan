"""Linear input dimensionality reducers."""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from robotorchan.reduction.base import InputReducer


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


