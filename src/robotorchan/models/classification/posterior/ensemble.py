"""Empirical posterior over classification probability vectors."""

from __future__ import annotations

import torch
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor


class ClassificationEnsemblePosterior(EnsemblePosterior):
    """Classification posterior represented by member probability vectors.

    The member axis is always the leading dimension and the class axis is
    always the final dimension. Intermediate dimensions are arbitrary model,
    batch, or candidate dimensions.
    """

    def __init__(self, probabilities: Tensor, weights: Tensor | None = None) -> None:
        if probabilities.ndim < 2:
            raise ValueError("probabilities must have shape members x ... x classes.")
        if probabilities.shape[0] < 2:
            raise ValueError("Classification ensembles require at least two members.")
        if probabilities.shape[-1] < 2:
            raise ValueError("The final dimension must contain at least two classes.")
        if not torch.is_floating_point(probabilities):
            raise ValueError("probabilities must use a floating-point dtype.")
        if not torch.isfinite(probabilities).all():
            raise ValueError("probabilities must be finite.")
        if torch.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("probabilities must lie in [0, 1].")
        sums = probabilities.sum(dim=-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
            raise ValueError("Each member probability vector must sum to one.")
        if weights is None:
            weights = torch.ones(
                probabilities.shape[0],
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
        if weights.ndim != 1 or weights.shape[0] != probabilities.shape[0]:
            raise ValueError("weights must have shape (num_members,).")
        if weights.dtype != probabilities.dtype or weights.device != probabilities.device:
            raise ValueError("weights must match probability dtype and device.")
        if not torch.isfinite(weights).all():
            raise ValueError("weights must be finite.")
        if torch.any(weights < 0.0):
            raise ValueError("weights must be non-negative.")
        if not bool(weights.sum() > 0.0):
            raise ValueError("At least one ensemble weight must be positive.")
        self._probabilities = probabilities
        # BoTorch stores ensemble members in the -3 axis, not the first axis.
        if probabilities.ndim == 2:
            values = probabilities.unsqueeze(-2)
        else:
            values = probabilities.movedim(0, -3)
        super().__init__(values=values, weights=weights)

    @property
    def probabilities(self) -> Tensor:
        """Return member probability vectors without copying."""
        return self._probabilities

    @property
    def num_members(self) -> int:
        """Return the number of empirical posterior members."""
        return self.ensemble_size

    @property
    def num_classes(self) -> int:
        """Return the number of predictive classes."""
        return self._probabilities.shape[-1]

    @property
    def variance(self) -> Tensor:
        """Return population probability variance across ensemble members.

        Unlike the Bessel-corrected sample variance of BoTorch EnsemblePosterior,
        weights represent discrete posterior probabilities, not frequency weights.
        This also supports a single member with nonzero weight without division
        by zero, preserving the existing robotorchan contract.
        """
        deviations = self.values - self.mean.unsqueeze(-3)
        return (deviations.square() * self.weights[..., None, None]).sum(dim=-3)

    @property
    def predictive_entropy(self) -> Tensor:
        """Return entropy of the ensemble-mean predictive distribution."""
        probabilities = self.mean
        tiny = torch.finfo(probabilities.dtype).tiny
        safe = probabilities.clamp_min(tiny)
        return -torch.special.xlogy(safe, safe).sum(dim=-1)

    @property
    def expected_class_entropy(self) -> Tensor:
        """Return mean entropy of member predictive distributions."""
        tiny = torch.finfo(self.dtype).tiny
        safe = self.values.clamp_min(tiny)
        entropy = -torch.special.xlogy(safe, safe).sum(dim=-1)
        return (entropy * self.weights[..., None]).sum(dim=-2)

    @property
    def mutual_information(self) -> Tensor:
        """Return BALD-style epistemic disagreement."""
        return (self.predictive_entropy - self.expected_class_entropy).clamp_min(0.0)

    @property
    def base_sample_shape(self) -> torch.Size:
        """Integer index samples have shape sample_shape + batch_shape."""
        return self.batch_shape

    @property
    def batch_range(self) -> tuple[int, int]:
        """The ensemble's t-batch axes, excluding the member and event axes."""
        return (0, len(self.batch_shape))

    def credible_interval(
        self,
        confidence_level: float = 0.95,
        *,
        num_samples: int = 4096,
    ) -> tuple[Tensor, Tensor]:
        """Return exact marginal equal-tailed quantiles of the discrete members.

        The optional sample count matches the GP probability-posterior API,
        but is unused: weighted quantiles here require no Monte Carlo sampling.
        Intervals describe disagreement between member mean probabilities,
        not the latent uncertainty within an individual GP.
        """
        del num_samples
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be strictly between zero and one.")
        values = self.probabilities.reshape(self.num_members, -1)
        sorted_values, indices = values.sort(dim=0)
        weight_columns = self.weights[:, None].expand_as(values)
        cumulative = weight_columns.gather(0, indices).cumsum(dim=0)
        tail = (1.0 - confidence_level) / 2.0
        lower_index = (cumulative >= tail).int().argmax(dim=0, keepdim=True)
        upper_index = (cumulative >= 1.0 - tail).int().argmax(dim=0, keepdim=True)
        event_shape = self.mean.shape
        lower = sorted_values.gather(0, lower_index).reshape(event_shape)
        upper = sorted_values.gather(0, upper_index).reshape(event_shape)
        return lower, upper


def make_classification_ensemble_posterior(
    probabilities: Tensor,
    weights: Tensor | None = None,
) -> ClassificationEnsemblePosterior:
    """Build a validated empirical posterior over class probabilities."""
    return ClassificationEnsemblePosterior(probabilities, weights=weights)
