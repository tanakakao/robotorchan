"""Single-task multiclass variational Gaussian process classifier."""

from __future__ import annotations

from copy import deepcopy

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import SoftmaxLikelihood
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

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
from robotorchan.models.classification.posterior.probability import (
    ClassificationProbabilityPosterior,
)
from robotorchan.models.classification.validation import validate_class_indices
from robotorchan.models.standard.variational import SingleTaskVariationalGP


class MulticlassSingleTaskGPClassifier(ClassificationModelMixin, SingleTaskVariationalGP):
    """Variational multiclass classifier with one latent GP per class."""

    likelihood_family = ClassificationLikelihoodFamily.CATEGORICAL
    latent_output_structure = LatentOutputStructure.PER_CLASS
    supports_mll = False

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
            likelihood=SoftmaxLikelihood(
                num_features=num_classes,
                num_classes=num_classes,
                mixing_weights=False,
            ),
            num_outputs=num_classes,
            inducing_points=inducing_points,
            learn_inducing_points=learn_inducing_points,
        )
        self._store_raw_tensor("train_Y", labels.detach().clone().long())
        self.class_models = nn.ModuleList(
            SingleTaskVariationalGP(
                train_X=train_X,
                num_outputs=1,
                covar_module=deepcopy(covar_module),
                inducing_points=inducing_points,
                learn_inducing_points=learn_inducing_points,
            )
            for _ in range(num_classes)
        )

    def make_mll(self, num_data: int | None = None) -> None:
        """Reject the inherited ELBO, which targets an unused upstream GP.

        Multiclass training uses independent class-wise latent GPs and a
        custom softmax variational objective in fit(). The inherited
        SingleTaskVariationalGP.make_mll() instead targets self.model,
        which is not used by fit() or posterior().
        """
        raise NotImplementedError(
            "Multiclass GP uses a class-wise variational objective in fit(); "
            "the inherited single-model VariationalELBO is not applicable."
        )

    def _log_kernel_prior(self) -> Tensor:
        """Optional covariance prior term for the class-wise variational objective."""
        return self.raw_train_X.new_zeros(())

    def fit(
        self,
        *,
        epochs: int = 60,
        lr: float = 0.03,
        batch_size: int | None = None,
    ) -> None:
        """Train categorical labels using a multitask latent distribution."""
        if epochs <= 0 or lr <= 0:
            raise ValueError("epochs and lr must be positive.")
        n = self.raw_train_X.shape[-2]
        if batch_size is None:
            batch_size = n
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        loader = DataLoader(
            TensorDataset(self.raw_train_X, self.raw_train_Y),
            batch_size=batch_size,
            shuffle=True,
        )
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.class_models.parameters(), lr=lr)
        for _ in range(epochs):
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                samples = torch.stack(
                    [gp.model(batch_X).rsample() for gp in self.class_models], dim=-1
                )
                log_probs = samples.log_softmax(dim=-1)
                log_likelihood = log_probs.gather(-1, batch_Y.long().reshape(-1, 1)).mean()
                kl = sum(
                    gp.model.variational_strategy.kl_divergence().sum() for gp in self.class_models
                )
                loss = -log_likelihood + kl / n - self._log_kernel_prior() / n
                loss.backward()
                optimizer.step()
        self.eval()
        self.likelihood.eval()

    def posterior(self, X: Tensor, **kwargs: object) -> GPyTorchPosterior:
        """Return independent class-wise latent GP posterior."""
        distributions = [gp.model(X) for gp in self.class_models]
        means = torch.stack([dist.mean for dist in distributions], dim=-2)
        covariances = torch.stack(
            [dist.lazy_covariance_matrix.to_dense() for dist in distributions],
            dim=-3,
        )
        batch_mvn = MultivariateNormal(means, covariances)
        return GPyTorchPosterior(MultitaskMultivariateNormal.from_batch_mvn(batch_mvn))

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
        return latent.softmax(dim=-1)

    def predict_proba(
        self,
        X: Tensor,
        *,
        num_samples: int = 256,
        **kwargs: object,
    ) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        samples = self.sample_class_probabilities(
            X, sample_shape=torch.Size([num_samples]), **kwargs
        )
        return samples.mean(dim=0)

    def probability_posterior(
        self,
        X: Tensor,
        *,
        num_samples: int = 1024,
        seed: int = 0,
        **kwargs: object,
    ) -> ClassificationProbabilityPosterior:
        """Return a class-probability Posterior with shared QMC moment samples.

        Softmax-normal moments have no general closed form. The mean and
        epistemic variance are estimated from the same seeded Sobol samples,
        preserving joint latent dependence across candidates and classes.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        latent = self.latent_posterior(X, **kwargs)
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([num_samples]),
            seed=seed,
        )
        probabilities = sampler(latent).softmax(dim=-1)
        return ClassificationProbabilityPosterior(
            latent,
            sample_transform=lambda values: values.softmax(dim=-1),
            mean=probabilities.mean(dim=0),
            variance=probabilities.var(dim=0, unbiased=False),
        )

    def probability_variance(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Estimate variance of each class probability over latent GP samples."""
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        samples = self.sample_class_probabilities(
            X, sample_shape=torch.Size([num_samples]), **kwargs
        )
        return samples.var(dim=0, unbiased=False)

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Estimate BALD information between labels and latent GP functions."""
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        samples = self.sample_class_probabilities(
            X, sample_shape=torch.Size([num_samples]), **kwargs
        )
        predictive = samples.mean(dim=0)
        predictive_entropy = -torch.special.xlogy(predictive, predictive).sum(dim=-1)
        conditional_entropy = -torch.special.xlogy(samples, samples).sum(dim=-1).mean(dim=0)
        return (predictive_entropy - conditional_entropy).clamp_min(0.0)

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


class MixedMulticlassSingleTaskGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass variational GP with native categorical covariance."""

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
        """Construct independent class GPs sharing the mixed-kernel specification."""
        covar_module = make_mixed_covar_module(
            input_dim=train_X.shape[-1],
            cat_dims=cat_dims,
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
        self.cat_dims = normalize_feature_dims(cat_dims, train_X.shape[-1], name="cat_dims")
