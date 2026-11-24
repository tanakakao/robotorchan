"""Class-conditional label-noise channels for multiclass and ordinal GP models.

Each row of transition T specifies P(observed=j | clean=i). The observed
label likelihood is trained directly, while clean and observed prediction
APIs remain separate. This first implementation uses fixed, known rates.
"""

from __future__ import annotations

import math

import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.likelihood import OrderedProbitLikelihood
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.classification.posterior.probability import (
    ClassificationProbabilityPosterior,
)


def _transition_matrix(
    num_classes: int,
    *,
    reference: Tensor,
    flip_probability: float,
    transition_matrix: Tensor | None,
) -> Tensor:
    """Validate P(observed label | clean label) for every true class."""
    if num_classes < 3:
        raise ValueError("num_classes must be at least 3.")
    if transition_matrix is not None:
        if flip_probability != 0.0:
            raise ValueError("Specify transition_matrix or flip_probability, not both.")
        matrix = torch.as_tensor(transition_matrix, device=reference.device, dtype=reference.dtype)
        if matrix.shape != (num_classes, num_classes):
            raise ValueError("transition_matrix must have shape [num_classes, num_classes].")
        if not torch.isfinite(matrix).all() or torch.any(matrix < 0) or torch.any(matrix > 1):
            raise ValueError("transition_matrix entries must be finite probabilities.")
        if not torch.allclose(
            matrix.sum(dim=-1),
            torch.ones(num_classes, device=matrix.device, dtype=matrix.dtype),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("transition_matrix rows must sum to one.")
        return matrix.detach().clone()
    if not isinstance(flip_probability, int | float) or not math.isfinite(flip_probability):
        raise ValueError("flip_probability must be finite.")
    if not 0 <= flip_probability < 1:
        raise ValueError("flip_probability must satisfy 0 <= value < 1.")
    off_diagonal = flip_probability / (num_classes - 1)
    matrix = torch.full(
        (num_classes, num_classes),
        off_diagonal,
        dtype=reference.dtype,
        device=reference.device,
    )
    matrix.fill_diagonal_(1.0 - flip_probability)
    return matrix


class LabelNoiseOrderedProbitLikelihood(OrderedProbitLikelihood):
    """Ordered-probit likelihood followed by a fixed label-transition channel."""

    def __init__(self, num_classes: int, transition_matrix: Tensor) -> None:
        super().__init__(num_classes)
        self.register_buffer("transition_matrix", transition_matrix.detach().clone())

    def clean_class_probabilities(self, latent: Tensor) -> Tensor:
        return super().class_probabilities(latent)

    def class_probabilities(self, latent: Tensor) -> Tensor:
        return self.clean_class_probabilities(latent) @ self.transition_matrix


class LabelNoiseOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordinal GP trained on noisy labels with distinct clean/observed predictions."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        flip_probability: float = 0.0,
        transition_matrix: Tensor | None = None,
    ) -> None:
        transition = _transition_matrix(
            num_classes,
            reference=train_X,
            flip_probability=flip_probability,
            transition_matrix=transition_matrix,
        )
        super().__init__(train_X, train_Y, num_classes=num_classes)
        self.likelihood = LabelNoiseOrderedProbitLikelihood(num_classes, transition).to(train_X)

    @property
    def transition_matrix(self) -> Tensor:
        return self.likelihood.transition_matrix

    def predict_clean_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Integrate the clean ordered-probit likelihood over the Gaussian latent."""
        return super().predict_proba(X, **kwargs)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return observed-label probabilities after the transition channel."""
        return self.predict_clean_proba(X, **kwargs) @ self.transition_matrix


class LabelNoiseMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Class-wise variational GP with fixed categorical label-confusion channel."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        flip_probability: float = 0.0,
        transition_matrix: Tensor | None = None,
    ) -> None:
        transition = _transition_matrix(
            num_classes,
            reference=train_X,
            flip_probability=flip_probability,
            transition_matrix=transition_matrix,
        )
        super().__init__(train_X, train_Y, num_classes=num_classes)
        self.register_buffer("transition_matrix", transition)

    def fit(
        self,
        *,
        epochs: int = 60,
        lr: float = 0.03,
        batch_size: int | None = None,
    ) -> None:
        """Maximize the noisy-label expected log likelihood minus variational KL."""
        if epochs <= 0 or lr <= 0:
            raise ValueError("epochs and lr must be positive.")
        n = self.raw_train_X.shape[-2]
        batch_size = n if batch_size is None else batch_size
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
                latent_samples = torch.stack(
                    [gp.model(batch_X).rsample() for gp in self.class_models],
                    dim=-1,
                )
                observed = latent_samples.softmax(dim=-1) @ self.transition_matrix
                selected = observed.gather(-1, batch_Y.long().reshape(-1, 1))
                log_likelihood = selected.clamp_min(
                    torch.finfo(selected.dtype).tiny
                ).log().mean()
                kl = sum(
                    gp.model.variational_strategy.kl_divergence().sum()
                    for gp in self.class_models
                )
                loss = -log_likelihood + kl / n
                loss.backward()
                optimizer.step()
        self.eval()
        self.likelihood.eval()

    def sample_clean_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        return self.sample_latent(X, sample_shape=sample_shape, **kwargs).softmax(dim=-1)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        return (
            self.sample_clean_class_probabilities(X, sample_shape=sample_shape, **kwargs)
            @ self.transition_matrix
        )

    def predict_clean_proba(self, X: Tensor, *, num_samples: int = 256) -> Tensor:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        return self.sample_clean_class_probabilities(
            X, sample_shape=torch.Size([num_samples])
        ).mean(dim=0)

    def probability_posterior(
        self,
        X: Tensor,
        *,
        num_samples: int = 1024,
        seed: int = 0,
    ) -> ClassificationProbabilityPosterior:
        """Observed-label epistemic moments under a shared Sobol sample."""
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        latent = self.latent_posterior(X)
        probabilities = (
            SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]), seed=seed)(latent)
            .softmax(dim=-1)
            @ self.transition_matrix
        )
        return ClassificationProbabilityPosterior(
            latent,
            sample_transform=lambda values: values.softmax(dim=-1) @ self.transition_matrix,
            mean=probabilities.mean(dim=0),
            variance=probabilities.var(dim=0, unbiased=False),
        )


def _contamination_transition(
    num_classes: int,
    reference: Tensor,
    contamination_probability: float,
    contaminant_distribution: Tensor | None,
) -> Tensor:
    """Construct a fixed class-independent contaminant-label process."""
    if not isinstance(contamination_probability, int | float) or not math.isfinite(
        contamination_probability
    ):
        raise ValueError("contamination_probability must be finite.")
    if not 0.0 < contamination_probability < 1.0:
        raise ValueError("contamination_probability must be strictly between 0 and 1.")
    base = (
        torch.full(
            (num_classes,), 1.0 / num_classes, device=reference.device, dtype=reference.dtype
        )
        if contaminant_distribution is None
        else torch.as_tensor(
            contaminant_distribution, device=reference.device, dtype=reference.dtype
        )
    )
    if base.shape != (num_classes,):
        raise ValueError("contaminant_distribution must have shape [num_classes].")
    if not torch.isfinite(base).all() or torch.any(base < 0) or torch.any(base > 1):
        raise ValueError("contaminant_distribution must contain finite probabilities.")
    if not torch.allclose(base.sum(), base.new_tensor(1.0), atol=1e-6, rtol=1e-5):
        raise ValueError("contaminant_distribution must sum to one.")
    identity = torch.eye(num_classes, device=reference.device, dtype=reference.dtype)
    return (1.0 - contamination_probability) * identity + contamination_probability * base


class ContaminatedMulticlassGPClassifier(LabelNoiseMulticlassGPClassifier):
    """Multiclass GP with an input-independent contaminant-label distribution."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        contamination_probability: float = 0.05,
        contaminant_distribution: Tensor | None = None,
    ) -> None:
        transition = _contamination_transition(
            num_classes, train_X, contamination_probability, contaminant_distribution
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, transition_matrix=transition)


class ContaminatedOrdinalGPClassifier(LabelNoiseOrdinalGPClassifier):
    """Ordinal GP with an input-independent contaminant-label distribution."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        contamination_probability: float = 0.05,
        contaminant_distribution: Tensor | None = None,
    ) -> None:
        transition = _contamination_transition(
            num_classes, train_X, contamination_probability, contaminant_distribution
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, transition_matrix=transition)
