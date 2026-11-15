"""Ordered-probit likelihood for ordinal Gaussian process classification."""

from __future__ import annotations

import torch
from gpytorch.likelihoods import Likelihood
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F


class OrderedProbitLikelihood(Likelihood):
    """Ordered-probit observation model with strictly increasing cutpoints.

    The first cutpoint is fixed to zero to remove the latent-location
    ambiguity. Remaining cutpoints are cumulative positive increments.
    """

    def __init__(self, num_classes: int) -> None:
        if num_classes < 3:
            raise ValueError("num_classes must be at least 3.")
        super().__init__()
        self.num_classes = num_classes
        self.raw_increments = nn.Parameter(torch.zeros(num_classes - 2))
        self.quadrature = GaussHermiteQuadrature1D(num_locs=20)

    @property
    def thresholds(self) -> Tensor:
        increments = F.softplus(self.raw_increments) + 1e-4
        zero = increments.new_zeros(1)
        return torch.cat((zero, increments.cumsum(dim=0)))

    def class_probabilities(self, latent: Tensor) -> Tensor:
        """Map latent values of shape [...] to ordinal probabilities [..., C]."""
        boundaries = self.thresholds
        z = boundaries - latent.unsqueeze(-1)
        cdf = torch.special.ndtr(z)
        first = cdf[..., :1]
        middle = cdf[..., 1:] - cdf[..., :-1]
        last = 1.0 - cdf[..., -1:]
        probabilities = torch.cat((first, middle, last), dim=-1)
        return probabilities.clamp_min(0.0) / probabilities.clamp_min(0.0).sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(probabilities.dtype).tiny)

    def forward(self, function_samples: Tensor, **kwargs: object) -> Categorical:
        return Categorical(probs=self.class_probabilities(function_samples))

    def expected_log_prob(
        self, target: Tensor, input: object, *params: object, **kwargs: object
    ) -> Tensor:
        """Evaluate expected ordinal log likelihood by Gaussian quadrature."""
        labels = target.long()

        def log_probability(latent: Tensor) -> Tensor:
            probabilities = self.class_probabilities(latent)
            expanded = labels.expand(probabilities.shape[:-1])
            selected = probabilities.gather(-1, expanded.unsqueeze(-1)).squeeze(-1)
            return selected.clamp_min(torch.finfo(selected.dtype).tiny).log()

        return self.quadrature(log_probability, input)
