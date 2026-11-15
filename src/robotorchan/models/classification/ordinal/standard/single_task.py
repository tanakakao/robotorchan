"""Single-task ordinal variational Gaussian process classifier."""

from __future__ import annotations

import torch
from gpytorch.kernels import Kernel
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch import Tensor
from torch.distributions import Categorical

from robotorchan.models.base import (
    ContinuousKernelFactory,
    make_mixed_covar_module,
    normalize_feature_dims,
)
from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.ordinal.likelihood import OrderedProbitLikelihood
from robotorchan.models.classification.posterior import ClassificationProbabilityPosterior
from robotorchan.models.classification.validation import validate_class_indices
from robotorchan.models.standard.variational import SingleTaskVariationalGP


class OrdinalSingleTaskGPClassifier(ClassificationModelMixin, SingleTaskVariationalGP):
    """Single-latent GP with an ordered-probit observation likelihood."""

    likelihood_family = ClassificationLikelihoodFamily.CATEGORICAL
    latent_output_structure = LatentOutputStructure.SINGLE

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        covar_module: Kernel | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        if num_classes < 3:
            raise ValueError("num_classes must be at least 3.")
        labels = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[-1] == 1 else train_Y
        validate_class_indices(labels, num_classes=num_classes)
        if labels.ndim != 1 or labels.shape[0] != train_X.shape[-2]:
            raise ValueError("train_Y must have shape [n] or [n, 1].")
        self._num_classes = num_classes
        super().__init__(
            train_X=train_X,
            train_Y=labels.to(dtype=train_X.dtype).unsqueeze(-1),
            likelihood=OrderedProbitLikelihood(num_classes),
            num_outputs=1,
            covar_module=covar_module,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )
        self._store_raw_tensor("train_Y", labels.detach().clone().long())
        self._probability_quadrature = GaussHermiteQuadrature1D(num_locs=64).to(train_X)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_labels(self) -> tuple[int, ...]:
        return tuple(range(self.num_classes))

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        latent = self.sample_latent(X, sample_shape=sample_shape, **kwargs)
        return self.likelihood.class_probabilities(latent.squeeze(-1))

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Integrate ordered-probit probabilities over the latent Gaussian."""
        posterior = self.latent_posterior(X, **kwargs)
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)
        thresholds = self.likelihood.thresholds
        scale = torch.sqrt(1.0 + variance.clamp_min(0.0)).unsqueeze(-1)
        cdf = torch.special.ndtr((thresholds - mean.unsqueeze(-1)) / scale)
        probabilities = torch.cat(
            (cdf[..., :1], cdf[..., 1:] - cdf[..., :-1], 1.0 - cdf[..., -1:]),
            dim=-1,
        ).clamp_min(0.0)
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    def probability_posterior(
        self,
        X: Tensor,
        **kwargs: object,
    ) -> ClassificationProbabilityPosterior:
        """Return ordinal class-probability uncertainty as a BoTorch Posterior.

        Mean probabilities are analytic; epistemic variances use one-dimensional
        Gaussian quadrature. Credible limits use marginal posterior samples,
        since probabilities of interior ordinal classes are not monotone in f.
        """
        latent = self.latent_posterior(X, **kwargs)
        mean = self.predict_proba(X, **kwargs)

        def probability_samples(values: Tensor) -> Tensor:
            return self.likelihood.class_probabilities(values.squeeze(-1))

        variances = []
        for index in range(self.num_classes):
            reference = mean[..., index]
            variance = self._probability_quadrature(
                lambda values, reference=reference, index=index: (
                    self.likelihood.class_probabilities(values)[..., index] - reference
                ).square(),
                latent.distribution,
            )
            variances.append(variance.clamp_min(0.0))
        variance = torch.stack(variances, dim=-1)
        return ClassificationProbabilityPosterior(
            latent,
            sample_transform=probability_samples,
            mean=mean,
            variance=variance,
        )

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Categorical:
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)


class MixedOrdinalSingleTaskGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordinal GP using the shared native mixed categorical covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        cat_dims: list[int],
        cont_kernel_factory: ContinuousKernelFactory | None = None,
        inducing_points: Tensor | int | None = None,
        learn_inducing_points: bool = True,
    ) -> None:
        normalized_cat_dims = normalize_feature_dims(cat_dims, train_X.shape[-1], name="cat_dims")
        covar_module = make_mixed_covar_module(
            input_dim=train_X.shape[-1],
            cat_dims=normalized_cat_dims,
            batch_shape=train_X.shape[:-2],
            cont_kernel_factory=cont_kernel_factory,
        )
        super().__init__(
            train_X,
            train_Y,
            num_classes=num_classes,
            covar_module=covar_module,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )
        self.cat_dims = tuple(normalized_cat_dims)
