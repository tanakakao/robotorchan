"""Empirical posterior over classification probability vectors."""

from __future__ import annotations

import torch
from torch import Tensor


class ClassificationEnsemblePosterior:
    """Classification posterior represented by member probability vectors.

    The member axis is always the leading dimension and the class axis is
    always the final dimension. Intermediate dimensions are arbitrary model,
    batch, or candidate dimensions.
    """

    def __init__(self, probabilities: Tensor) -> None:
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
        self._probabilities = probabilities

    @property
    def probabilities(self) -> Tensor:
        """Return member probability vectors without copying."""
        return self._probabilities

    @property
    def num_members(self) -> int:
        """Return the number of empirical posterior members."""
        return self._probabilities.shape[0]

    @property
    def num_classes(self) -> int:
        """Return the number of predictive classes."""
        return self._probabilities.shape[-1]

    @property
    def mean(self) -> Tensor:
        """Return the ensemble-mean class probability."""
        return self._probabilities.mean(dim=0)

    @property
    def variance(self) -> Tensor:
        """Return epistemic class-probability variance across members."""
        return self._probabilities.var(dim=0, unbiased=False)

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
        tiny = torch.finfo(self._probabilities.dtype).tiny
        safe = self._probabilities.clamp_min(tiny)
        entropy = -torch.special.xlogy(safe, safe).sum(dim=-1)
        return entropy.mean(dim=0)

    @property
    def mutual_information(self) -> Tensor:
        """Return BALD-style epistemic disagreement."""
        return (self.predictive_entropy - self.expected_class_entropy).clamp_min(0.0)

    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        """Sample member probability vectors from the empirical posterior."""
        resolved_shape = torch.Size([1]) if sample_shape is None else torch.Size(sample_shape)
        if not resolved_shape:
            raise ValueError("sample_shape must contain at least one sample dimension.")
        num_samples = resolved_shape.numel()
        indices = torch.randint(
            self.num_members,
            (num_samples,),
            device=self._probabilities.device,
        )
        samples = self._probabilities.index_select(0, indices)
        return samples.reshape(*resolved_shape, *self._probabilities.shape[1:])


def make_classification_ensemble_posterior(
    probabilities: Tensor,
) -> ClassificationEnsemblePosterior:
    """Build a validated empirical posterior over class probabilities."""
    return ClassificationEnsemblePosterior(probabilities)
