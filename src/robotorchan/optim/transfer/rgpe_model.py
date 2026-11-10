"""Moment-matched posterior ensemble for ranking-weighted transfer BO."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor


class RGPEModel(Model):
    """Combine scalar-output GP posteriors with fixed RGPE weights.

    The mixture is approximated by a Gaussian matching its first two
    moments. This preserves within-model covariance and adds uncertainty
    from disagreement between models. Weights are obtained separately
    using :func:`rgpe_weights`.
    """

    def __init__(
        self,
        source_models: Sequence[Model],
        target_model: Model,
        weights: Tensor,
    ) -> None:
        super().__init__()
        models = [*source_models, target_model]
        if weights.ndim != 1 or weights.numel() != len(models):
            raise ValueError("weights must have one entry per model")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("weights must be finite and nonnegative")
        if not bool(torch.isclose(weights.sum(), weights.new_tensor(1.0))):
            raise ValueError("weights must sum to one")
        if any(model.num_outputs != 1 for model in models):
            raise ValueError("RGPEModel supports scalar-output models only")
        self.models = torch.nn.ModuleList(models)
        self.register_buffer("weights", weights.detach().clone())

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ) -> GPyTorchPosterior:
        """Return a moment-matched Gaussian posterior for the ensemble."""
        if output_indices is not None and output_indices != [0]:
            raise ValueError("Only the single output index 0 is supported")
        means = []
        covariances = []
        for model in self.models:
            posterior = model.posterior(X, observation_noise=observation_noise)
            means.append(posterior.mean.squeeze(-1))
            covariance = posterior.mvn.covariance_matrix
            covariances.append(covariance)
        means_tensor = torch.stack(means)
        covariance_tensor = torch.stack(covariances)
        weights = self.weights.to(dtype=means_tensor.dtype, device=means_tensor.device)
        weight_shape = (-1,) + (1,) * (means_tensor.ndim - 1)
        weighted_mean = (weights.reshape(weight_shape) * means_tensor).sum(dim=0)
        second_moment = (
            weights.reshape((-1,) + (1,) * (covariance_tensor.ndim - 1))
            * covariance_tensor
        ).sum(dim=0)
        deviations = means_tensor - weighted_mean.unsqueeze(0)
        between = torch.einsum("m,m...i,m...j->...ij", weights, deviations, deviations)
        covariance = second_moment + between
        posterior = GPyTorchPosterior(MultivariateNormal(weighted_mean, covariance))
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior
