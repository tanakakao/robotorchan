"""Single-task binary classification surrogate models."""

from __future__ import annotations

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.variational import VariationalStrategy, _VariationalDistribution, _VariationalStrategy
from torch import Tensor, nn
from torch.distributions import Bernoulli

from robotorchan.models.base import (
    ContinuousKernelFactory,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.models.classification.binary.base import BinaryClassificationMixin
from robotorchan.models.classification.binary.validation import validate_binary_labels
from robotorchan.models.standard.variational import SingleTaskVariationalGP


class BinarySingleTaskGPClassifier(BinaryClassificationMixin, SingleTaskVariationalGP):
    """Binary GP classifier backed by BoTorch variational GP infrastructure."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        learn_inducing_points: bool = True,
        covar_module: Kernel | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a binary variational GP with Bernoulli likelihood."""
        validate_binary_labels(train_Y)
        if train_X.shape[-2] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of rows.")
        raw_train_Y = train_Y.detach().clone()
        model_train_Y = train_Y.unsqueeze(-1) if train_Y.ndim == train_X.ndim - 1 else train_Y
        model_train_Y = model_train_Y.to(dtype=train_X.dtype, device=train_X.device)
        super().__init__(
            train_X=train_X,
            train_Y=model_train_Y,
            likelihood=BernoulliLikelihood(),
            num_outputs=1,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            outcome_transform=None,
            input_transform=input_transform,
        )
        self._store_raw_tensor("train_Y", raw_train_Y)
        self._probability_quadrature = GaussHermiteQuadrature1D(num_locs=128).to(train_X)

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load variational classifier state without Gaussian target extraction."""
        result = nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)
        self.model.variational_strategy.variational_params_initialized.fill_(1)
        self.apply(clear_cache_hook)
        return result

    def make_mll(self, num_data: int | None = None) -> VariationalELBO:
        """Construct the Bernoulli variational evidence lower bound.

        Args:
            num_data: Total number of binary observations represented by the
                ELBO. Defaults to the number of caller-supplied training rows.

        Returns:
            Variational ELBO bound to this classifier likelihood and latent GP.
        """
        return super().make_mll(num_data=num_data)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Bernoulli:
        """Return Bernoulli predictions after integrating latent uncertainty."""
        latent = self.latent_posterior(X, **kwargs)
        return self.likelihood(latent.distribution)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return ``[P(y=0), P(y=1)]`` along the final dimension."""
        positive = self.predictive_distribution(X, **kwargs).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Draw class probabilities by mapping latent posterior samples."""
        latent_samples = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        positive = self.likelihood.forward(latent_samples).probs
        if positive.shape[-1:] == (1,):
            positive = positive.squeeze(-1)
        return torch.stack((1.0 - positive, positive), dim=-1)

    def _positive_probability_from_latent(self, values: Tensor, X: Tensor) -> Tensor:
        """Evaluate the observed-label link for supplied latent values."""
        del X
        return self.likelihood.forward(values).probs

    def probability_variance(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        method: str = "quadrature",
        **kwargs: object,
    ) -> Tensor:
        """Return epistemic class-probability variance using Gaussian quadrature.

        The default integrates the squared likelihood probability over the
        marginal latent Gaussian. Unlike Bernoulli predictive variance, this
        measures uncertainty about the probability itself. The num_samples
        argument is retained for compatibility and controls method="mc" only.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        if method == "mc":
            return super().probability_variance(X, num_samples=num_samples, **kwargs)
        if method != "quadrature":
            raise ValueError("method must be 'quadrature' or 'mc'.")

        latent = self.latent_posterior(X, **kwargs)
        mean_positive = self.predictive_distribution(X, **kwargs).probs
        variance = self._probability_quadrature(
            lambda values: (
                self._positive_probability_from_latent(values, X) - mean_positive
            ).square(),
            latent.distribution,
        ).clamp_min(0.0)
        if variance.shape[-1:] == (1,):
            variance = variance.squeeze(-1)
        return torch.stack((variance, variance), dim=-1)

    def expected_class_entropy(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        method: str = "quadrature",
        **kwargs: object,
    ) -> Tensor:
        """Integrate conditional label entropy over the latent Gaussian."""
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        if method == "mc":
            return super().expected_class_entropy(X, num_samples=num_samples, **kwargs)
        if method != "quadrature":
            raise ValueError("method must be 'quadrature' or 'mc'.")

        latent = self.latent_posterior(X, **kwargs)

        def entropy(values: Tensor) -> Tensor:
            positive = self._positive_probability_from_latent(values, X)
            # Bernoulli likelihoods can saturate exactly at 0 or 1.
            # Bound both sides to avoid undefined xlogy gradients at those limits.
            eps = torch.finfo(positive.dtype).eps
            safe = positive.clamp(min=eps, max=1.0 - eps)
            return -torch.special.xlogy(safe, safe) - torch.special.xlogy(1.0 - safe, 1.0 - safe)

        result = self._probability_quadrature(entropy, latent.distribution)
        return result.squeeze(-1) if result.shape[-1:] == (1,) else result

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        method: str = "quadrature",
        **kwargs: object,
    ) -> Tensor:
        """Return BALD from predictive entropy minus conditional entropy."""
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        if method == "mc":
            return super().mutual_information(X, num_samples=num_samples, **kwargs)
        if method != "quadrature":
            raise ValueError("method must be 'quadrature' or 'mc'.")

        predictive = self.predictive_entropy(X, **kwargs)
        conditional = self.expected_class_entropy(X, method="quadrature", **kwargs)
        return (predictive - conditional).clamp_min(0.0)

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return Bernoulli variance for each class probability."""
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return Bernoulli predictive entropy for each input point."""
        probabilities = self.predict_proba(X, **kwargs)
        terms = torch.special.xlogy(probabilities, probabilities)
        return -terms.sum(dim=-1)

    def predict_class(
        self,
        X: Tensor,
        *,
        threshold: float = 0.5,
        **kwargs: object,
    ) -> Tensor:
        """Return binary class labels using the requested probability threshold."""
        if not isinstance(threshold, int | float):
            raise TypeError("threshold must be a real number.")
        if not torch.isfinite(torch.tensor(threshold)):
            raise ValueError("threshold must be finite.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        positive = self.predict_proba(X, **kwargs)[..., 1]
        return torch.where(
            positive >= threshold,
            torch.ones_like(positive, dtype=torch.long),
            torch.zeros_like(positive, dtype=torch.long),
        )


class MixedBinarySingleTaskGPClassifier(BinarySingleTaskGPClassifier):
    """Binary variational GP classifier for native mixed continuous/categorical inputs."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        learn_inducing_points: bool = True,
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        """Initialize a binary classifier with native categorical covariance."""
        covar_module = make_mixed_covar_module(
            input_dim=train_X.shape[-1],
            cat_dims=cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.cat_dims = normalize_feature_dims(
            cat_dims,
            train_X.shape[-1],
            name="cat_dims",
        )
