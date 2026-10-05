"""Bootstrap ensemble contract for binary classification."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

from robotorchan.models.classification.binary.non_gp.base import NonGPBinaryClassificationMixin
from robotorchan.models.classification.binary.validation import validate_binary_labels
from robotorchan.models.classification.posterior import (
    ClassificationEnsemblePosterior,
    make_classification_ensemble_posterior,
)


class BootstrapBinaryClassificationEnsemble(NonGPBinaryClassificationMixin, nn.Module):
    """Empirical probability posterior built from complete bootstrap classifiers."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        n_members: int = 16,
        random_state: int | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        super().__init__()
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        labels = train_Y.squeeze(-1) if train_Y.ndim == 2 else train_Y
        validate_binary_labels(labels)
        if labels.ndim != 1 or labels.shape[0] != train_X.shape[0]:
            raise ValueError("train_Y must contain one binary label per training row.")
        if n_members < 2:
            raise ValueError("n_members must be at least 2.")
        self.register_buffer("raw_train_X", train_X.clone())
        self.register_buffer("raw_train_Y", labels.clone())
        self.n_members = n_members
        self.random_state = random_state
        self.estimator_kwargs = dict(estimator_kwargs)
        self._members: list[Any] = []
        self._is_fitted = False

    @abstractmethod
    def _make_estimator(self, member_index: int) -> Any:
        """Construct one complete classifier."""

    @property
    def is_fitted(self) -> bool:
        """Whether every bootstrap member has been fitted."""
        return self._is_fitted

    def fit(self) -> None:
        """Fit complete classifiers on independent bootstrap resamples."""
        X = self.raw_train_X.detach().cpu().numpy()
        y = self.raw_train_Y.detach().cpu().numpy()
        generator = torch.Generator()
        if self.random_state is None:
            generator.seed()
        else:
            generator.manual_seed(self.random_state)

        self._members = []
        for member_index in range(self.n_members):
            for _ in range(100):
                indices = torch.randint(len(X), (len(X),), generator=generator).numpy()
                member_y = y[indices]
                if len(set(member_y.tolist())) == self.num_classes:
                    break
            else:
                raise RuntimeError("Could not draw a bootstrap sample containing both classes.")
            member = self._make_estimator(member_index)
            member.fit(X[indices], member_y)
            self._members.append(member)
        self._is_fitted = True

    def _member_probabilities(self, X: Tensor) -> Tensor:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before probability prediction.")
        original_shape = X.shape[:-1]
        flat_X = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        probabilities = [
            torch.as_tensor(member.predict_proba(flat_X), dtype=X.dtype, device=X.device)
            for member in self._members
        ]
        values = torch.stack(probabilities)
        return values.reshape(self.n_members, *original_shape, self.num_classes)

    def probability_posterior(self, X: Tensor, **kwargs: object) -> ClassificationEnsemblePosterior:
        """Return the empirical posterior over member class probabilities."""
        del kwargs
        return make_classification_ensemble_posterior(self._member_probabilities(X))

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return the ensemble-mean predictive class probability."""
        return self.probability_posterior(X, **kwargs).mean

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Sample the empirical posterior by resampling complete ensemble members."""
        return self.probability_posterior(X, **kwargs).rsample(sample_shape)

    def probability_variance(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return epistemic variance across complete bootstrap members."""
        del num_samples
        return self.probability_posterior(X, **kwargs).variance

    def expected_class_entropy(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return mean conditional entropy across bootstrap members."""
        del num_samples
        return self.probability_posterior(X, **kwargs).expected_class_entropy

    def mutual_information(
        self,
        X: Tensor,
        *,
        num_samples: int = 128,
        **kwargs: object,
    ) -> Tensor:
        """Return BALD-style disagreement from the empirical member posterior."""
        del num_samples
        return self.probability_posterior(X, **kwargs).mutual_information
