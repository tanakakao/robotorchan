"""Output-space reducers for high-dimensional GP outcomes."""

from __future__ import annotations

import torch
from botorch.posteriors import Posterior
from torch import Tensor

from robotorchan.models.reduction import OutputReducer


class OutputPCAReducer(OutputReducer):
    """Principal-component reduction for high-dimensional outcomes.

    The reducer learns a linear basis in the original outcome space and supports
    deterministic forward and inverse transforms. Posterior uncertainty
    propagation is intentionally implemented in the next phase.

    Args:
        n_components: Number of principal components to retain.
        center: Whether to subtract the training-outcome mean before projection.
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

    def _inverse_transform_2d(self, Y: Tensor) -> Tensor:
        assert self.mean is not None
        assert self.components is not None
        restored = Y @ self.components.transpose(-2, -1)
        if self.center:
            restored = restored + self.mean
        return restored

    def restore_posterior(self, posterior: Posterior) -> Posterior:
        del posterior
        raise NotImplementedError(
            "OutputPCAReducer posterior restoration is implemented in Phase 5."
        )


class OutputPLSReducer(OutputReducer):
    """Supervised PLS-style reduction for high-dimensional outcomes.

    The original outcomes are projected onto directions with strong
    cross-covariance with the paired predictor tensor. A least-squares decoder
    learned from the latent training scores maps reduced values back to the
    original outcome space.

    Args:
        n_components: Number of supervised latent output components.
        center: Whether to center outcomes and predictors during fitting.
        eps: Numerical threshold for degenerate components.
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
        self.register_buffer("output_mean", None)
        self.register_buffer("predictor_mean", None)
        self.register_buffer("weights", None)
        self.register_buffer("loadings", None)
        self.register_buffer("rotation", None)
        self.register_buffer("decoder", None)

    def _fit_2d(self, X: Tensor, Y: Tensor | None) -> int:
        if Y is None:
            raise ValueError("OutputPLSReducer requires paired predictor data X.")
        if Y.ndim != 2:
            raise ValueError("OutputPLSReducer expects paired predictors with shape [n, d].")

        max_components = min(X.shape[0] - 1, X.shape[-1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds the PLS limit {max_components}."
            )

        output_mean = X.mean(dim=0) if self.center else torch.zeros_like(X[0])
        predictor_mean = Y.mean(dim=0) if self.center else torch.zeros_like(Y[0])
        output_centered = X - output_mean
        predictor_centered = Y - predictor_mean
        output_residual = output_centered.clone()
        predictor_residual = predictor_centered.clone()

        weights: list[Tensor] = []
        loadings: list[Tensor] = []
        for _ in range(self.n_components):
            cross_covariance = output_residual.transpose(-2, -1) @ predictor_residual
            left_vectors, singular_values, _ = torch.linalg.svd(
                cross_covariance,
                full_matrices=False,
            )
            if singular_values.numel() == 0 or singular_values[0] <= self.eps:
                raise ValueError("PLS could not extract the requested number of components.")

            weight = left_vectors[:, 0]
            score = output_residual @ weight
            score_norm = score.square().sum()
            if score_norm <= self.eps:
                raise ValueError("PLS encountered a degenerate latent score.")

            output_loading = (output_residual.transpose(-2, -1) @ score) / score_norm
            predictor_loading = (predictor_residual.transpose(-2, -1) @ score) / score_norm

            output_residual = output_residual - score.unsqueeze(-1) * output_loading.unsqueeze(0)
            predictor_residual = (
                predictor_residual - score.unsqueeze(-1) * predictor_loading.unsqueeze(0)
            )
            weights.append(weight)
            loadings.append(output_loading)

        weight_matrix = torch.stack(weights, dim=-1)
        loading_matrix = torch.stack(loadings, dim=-1)
        inner = loading_matrix.transpose(-2, -1) @ weight_matrix
        rotation = weight_matrix @ torch.linalg.pinv(inner)
        latent_scores = output_centered @ rotation
        decoder = torch.linalg.pinv(latent_scores) @ output_centered

        self.output_mean = output_mean
        self.predictor_mean = predictor_mean
        self.weights = weight_matrix
        self.loadings = loading_matrix
        self.rotation = rotation
        self.decoder = decoder
        return self.n_components

    def _transform_2d(self, X: Tensor) -> Tensor:
        assert self.output_mean is not None
        assert self.rotation is not None
        centered = X - self.output_mean if self.center else X
        return centered @ self.rotation

    def _inverse_transform_2d(self, Y: Tensor) -> Tensor:
        assert self.output_mean is not None
        assert self.decoder is not None
        restored = Y @ self.decoder
        if self.center:
            restored = restored + self.output_mean
        return restored

    def restore_posterior(self, posterior: Posterior) -> Posterior:
        del posterior
        raise NotImplementedError(
            "OutputPLSReducer posterior restoration is implemented in Phase 5."
        )
