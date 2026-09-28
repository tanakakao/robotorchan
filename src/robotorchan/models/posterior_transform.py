"""PosteriorTransform helpers for posterior types not handled by BoTorch utilities."""

from __future__ import annotations

import torch
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.posteriors import Posterior
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor

from robotorchan.reduction.output import LinearOutputPosterior


def _scalarized_values(
    values: Tensor,
    transform: ScalarizedPosteriorTransform,
) -> Tensor:
    """Apply an affine output scalarization to sample values."""
    weights = transform.weights.to(values)
    if values.shape[-1] != weights.numel():
        raise RuntimeError(
            f"Output dimension {values.shape[-1]} does not match "
            f"{weights.numel()} scalarization weights."
        )
    offset = transform.offset.to(values)
    return offset + (values * weights).sum(dim=-1, keepdim=True)


def scalarize_ensemble_posterior(
    posterior: EnsemblePosterior,
    transform: ScalarizedPosteriorTransform,
) -> EnsemblePosterior:
    """Scalarize every ensemble member while preserving empirical uncertainty."""
    return EnsemblePosterior(values=_scalarized_values(posterior.values, transform))


def scalarize_linear_output_posterior(
    posterior: Posterior,
    transform: ScalarizedPosteriorTransform,
) -> EnsemblePosterior:
    """Scalarize a restored linear-output posterior through latent samples.

    The restored posterior does not expose a GPyTorch distribution. Sampling
    preserves the fitted reducer's full linear output map, so an empirical
    posterior keeps the cross-output dependence needed by scalarization.
    """
    if not isinstance(posterior, LinearOutputPosterior):
        raise TypeError("Expected a LinearOutputPosterior.")
    sample_shape = torch.Size([256])
    samples = posterior.rsample(sample_shape=sample_shape)
    values = _scalarized_values(samples, transform)
    ensemble_dim = values.ndim - 3
    values = values.movedim(0, ensemble_dim)
    return EnsemblePosterior(values=values)
