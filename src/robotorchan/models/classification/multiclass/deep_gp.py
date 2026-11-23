"""Multiclass DeepGP classification using independent stochastic latent hierarchies."""

from __future__ import annotations

from collections.abc import Sequence

import gpytorch
import torch
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor, nn
from torch.distributions import Categorical

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.posterior.probability import (
    ClassificationProbabilityPosterior,
)
from robotorchan.models.classification.validation import validate_class_indices
from robotorchan.models.expressive.deep_gp import SingleTaskDeepGP


class MulticlassSingleTaskDeepGPClassifier(ClassificationModelMixin, Model):
    """Class-wise variational DeepGPs with a categorical softmax likelihood.

    DeepGP latent posteriors are not Gaussian. An empirical EnsemblePosterior
    preserves the class and joint-candidate axes without Gaussian assumptions.
    The training objective approximates expected categorical log likelihood
    and subtracts the variational KL from all DeepGP layers.
    """

    likelihood_family = ClassificationLikelihoodFamily.CATEGORICAL
    latent_output_structure = LatentOutputStructure.PER_CLASS
    supports_fit = True
    supports_mll = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        hidden_dims: Sequence[int] = (4,),
        num_inducing: int = 16,
        standardize_inputs: bool = True,
        random_state: int = 0,
        posterior_samples: int = 64,
    ) -> None:
        super().__init__()
        if num_classes < 3:
            raise ValueError("num_classes must be at least 3.")
        labels = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[-1] == 1 else train_Y
        validate_class_indices(labels, num_classes=num_classes)
        if train_X.ndim != 2 or labels.ndim != 1 or labels.shape[0] != train_X.shape[0]:
            raise ValueError("train_X must be [n,d] and train_Y must be [n] or [n,1].")
        if posterior_samples < 2:
            raise ValueError("posterior_samples must be at least 2.")
        self._num_classes = num_classes
        self.posterior_samples = posterior_samples
        self.register_buffer("train_X", train_X.detach().clone())
        self.register_buffer("train_Y", labels.detach().clone().long())
        self.class_models = nn.ModuleList(
            SingleTaskDeepGP(
                train_X=train_X,
                train_Y=labels.to(train_X).unsqueeze(-1),
                hidden_dims=hidden_dims,
                num_inducing=num_inducing,
                standardize_inputs=standardize_inputs,
                random_state=random_state + index,
                posterior_samples=posterior_samples,
            )
            for index in range(num_classes)
        )

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_outputs(self) -> int:
        return self.num_classes

    @property
    def class_labels(self) -> tuple[int, ...]:
        return tuple(range(self.num_classes))

    def fit(
        self,
        *,
        epochs: int = 100,
        lr: float = 0.01,
        num_likelihood_samples: int = 8,
    ) -> None:
        if epochs <= 0 or lr <= 0 or num_likelihood_samples < 2:
            raise ValueError("epochs, lr must be positive and likelihood samples >= 2.")
        optimizer = torch.optim.Adam(self.class_models.parameters(), lr=lr)
        self.train()
        n = self.train_X.shape[0]
        for _ in range(epochs):
            optimizer.zero_grad()
            with gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
                latent = torch.stack(
                    [gp(self.train_X).rsample() for gp in self.class_models], dim=-1
                )
            log_prob = latent.log_softmax(dim=-1)
            labels = self.train_Y.view(1, n, 1).expand(num_likelihood_samples, n, 1)
            log_likelihood = log_prob.gather(dim=-1, index=labels).mean()
            kl = sum(
                (
                    sum(
                        layer.variational_strategy.kl_divergence().sum()
                        for layer in (*gp.hidden_layers, gp.output_layer)
                    )
                    for gp in self.class_models
                ),
                self.train_X.new_zeros(()),
            )
            loss = -log_likelihood + kl / n
            loss.backward()
            optimizer.step()
        self.eval()

    def posterior(
        self,
        X: Tensor,
        *,
        num_samples: int | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ) -> EnsemblePosterior:
        if observation_noise is not False:
            raise NotImplementedError("DeepGP class-latent posterior has no observation noise.")
        resolved = self.posterior_samples if num_samples is None else int(num_samples)
        if resolved < 2:
            raise ValueError("num_samples must be at least 2.")
        trajectories = torch.stack(
            [
                gp.posterior(X, num_samples=resolved).rsample(torch.Size([resolved])).squeeze(-1)
                for gp in self.class_models
            ],
            dim=-1,
        )
        posterior = EnsemblePosterior(values=trajectories.movedim(0, -3))
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        *,
        num_samples: int | None = None,
    ) -> Tensor:
        latent = self.latent_posterior(X, num_samples=num_samples)
        resolved_shape = torch.Size() if sample_shape is None else sample_shape
        return latent.rsample(resolved_shape).softmax(dim=-1)

    def predict_proba(self, X: Tensor, *, num_samples: int | None = None) -> Tensor:
        resolved = self.posterior_samples if num_samples is None else int(num_samples)
        if resolved < 2:
            raise ValueError("num_samples must be at least 2.")
        samples = self.sample_class_probabilities(
            X, sample_shape=torch.Size([resolved]), num_samples=resolved
        )
        return samples.mean(dim=0)

    def probability_posterior(
        self,
        X: Tensor,
        *,
        num_samples: int | None = None,
    ) -> ClassificationProbabilityPosterior:
        resolved = self.posterior_samples if num_samples is None else int(num_samples)
        latent = self.latent_posterior(X, num_samples=resolved)
        samples = latent.rsample(torch.Size([resolved])).softmax(dim=-1)
        return ClassificationProbabilityPosterior(
            latent,
            sample_transform=lambda values: values.softmax(dim=-1),
            mean=samples.mean(dim=0),
            variance=samples.var(dim=0, unbiased=False),
        )

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)

    def predictive_distribution(self, X: Tensor, **kwargs: object) -> Categorical:
        return Categorical(probs=self.predict_proba(X, **kwargs))

    def predictive_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return probabilities * (1.0 - probabilities)

    def predictive_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        probabilities = self.predict_proba(X, **kwargs)
        return -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
