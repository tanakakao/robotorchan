"""Scikit-learn adapters for multiclass and ordinal-label classification.

These are discriminative estimators, not latent Gaussian process models.
Ordinal labels are valid labels here; the underlying estimator does not learn
ordinal thresholds or an ordered-probit likelihood.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from robotorchan.models.classification.base import (
    ClassificationLikelihoodFamily,
    ClassificationModelMixin,
    LatentOutputStructure,
)
from robotorchan.models.classification.validation import validate_class_indices

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
except ImportError:  # pragma: no cover
    ExtraTreesClassifier = None
    GradientBoostingClassifier = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None


class SklearnMulticlassClassifier(ClassificationModelMixin, nn.Module):
    """Fitted scikit-learn classifier with probability-only robotorchan API.

    The class axis is always 0..num_classes-1, even when some classes were
    absent from the estimator's training sample.
    """

    estimator_class: ClassVar[type | None] = None
    is_non_gp_classification: ClassVar[bool] = True
    has_latent_posterior: ClassVar[bool] = False
    likelihood_family = ClassificationLikelihoodFamily.CATEGORICAL
    latent_output_structure = LatentOutputStructure.CUSTOM

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        num_classes: int,
        **estimator_kwargs: Any,
    ) -> None:
        super().__init__()
        if self.estimator_class is None:
            raise ImportError("This classifier requires scikit-learn.")
        if num_classes < 3:
            raise ValueError("num_classes must be at least 3.")
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        labels = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[-1] == 1 else train_Y
        validate_class_indices(labels, num_classes=num_classes)
        if labels.ndim != 1 or labels.shape[0] != train_X.shape[0]:
            raise ValueError("train_Y must have shape [n] or [n, 1].")
        self._num_classes = num_classes
        self.register_buffer("raw_train_X", train_X.detach().clone())
        self.register_buffer("raw_train_Y", labels.detach().clone().long())
        self._estimator = self.estimator_class(**estimator_kwargs)
        self._is_fitted = False

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def class_labels(self) -> tuple[int, ...]:
        return tuple(range(self.num_classes))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self) -> None:
        """Train on raw input rows; no pseudo-GP latent posterior is created."""
        self._estimator.fit(
            self.raw_train_X.detach().cpu().numpy(),
            self.raw_train_Y.detach().cpu().numpy(),
        )
        self._is_fitted = True

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        if X.ndim < 2 or X.shape[-1] != self.raw_train_X.shape[-1]:
            raise ValueError("Candidate X must have shape [..., d].")
        flat = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        probabilities = self._estimator.predict_proba(flat)
        result = X.new_zeros((flat.shape[0], self.num_classes))
        classes = torch.as_tensor(self._estimator.classes_, dtype=torch.long, device=X.device)
        result[:, classes] = torch.as_tensor(
            probabilities, dtype=X.dtype, device=X.device
        )
        return result.reshape(*X.shape[:-1], self.num_classes)

    def latent_posterior(self, X: Tensor, **kwargs: object):
        del X, kwargs
        raise NotImplementedError("Non-GP classification has no latent GP posterior.")

    def sample_latent(
        self, X: Tensor, sample_shape: torch.Size | None = None, **kwargs: object
    ) -> Tensor:
        del X, sample_shape, kwargs
        raise NotImplementedError("Non-GP classification has no latent GP samples.")

    def sample_class_probabilities(
        self, X: Tensor, sample_shape: torch.Size | None = None, **kwargs: object
    ) -> Tensor:
        del X, sample_shape, kwargs
        raise NotImplementedError(
            "One deterministic non-GP estimator has no epistemic probability posterior; "
            "use a classification ensemble instead."
        )

    def probability_posterior(self, X: Tensor, **kwargs: object):
        del X, kwargs
        raise NotImplementedError(
            "One deterministic estimator has no probability posterior; use an ensemble."
        )

    def probability_variance(self, X: Tensor, **kwargs: object) -> Tensor:
        del X, kwargs
        raise NotImplementedError("Single non-GP estimator has no epistemic variance.")

    def expected_class_entropy(self, X: Tensor, **kwargs: object) -> Tensor:
        del X, kwargs
        raise NotImplementedError("Single non-GP estimator has no epistemic class entropy.")

    def mutual_information(self, X: Tensor, **kwargs: object) -> Tensor:
        del X, kwargs
        raise NotImplementedError("Single non-GP estimator has no epistemic BALD.")

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


class RandomForestMulticlassClassifier(SklearnMulticlassClassifier):
    estimator_class = RandomForestClassifier


class ExtraTreesMulticlassClassifier(SklearnMulticlassClassifier):
    estimator_class = ExtraTreesClassifier


class GradientBoostingMulticlassClassifier(SklearnMulticlassClassifier):
    estimator_class = GradientBoostingClassifier


class HistGradientBoostingMulticlassClassifier(SklearnMulticlassClassifier):
    estimator_class = HistGradientBoostingClassifier


class RandomForestOrdinalClassifier(RandomForestMulticlassClassifier):
    """Tree-based ordered-category prediction; does not impose ordinal thresholds."""


class ExtraTreesOrdinalClassifier(ExtraTreesMulticlassClassifier):
    """Tree-based ordered-category prediction; no ordinal likelihood assumed."""


class GradientBoostingOrdinalClassifier(GradientBoostingMulticlassClassifier):
    """Boosting ordered-category prediction without ordinal likelihood."""


class HistGradientBoostingOrdinalClassifier(HistGradientBoostingMulticlassClassifier):
    """Histogram boosting ordered-category prediction without ordinal likelihood."""
