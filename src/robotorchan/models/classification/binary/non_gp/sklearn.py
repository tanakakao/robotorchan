"""Scikit-learn binary classifier adapters."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from robotorchan.models.classification.binary.non_gp.base import (
    NonGPBinaryClassificationMixin,
)
from robotorchan.models.classification.binary.validation import validate_binary_labels

try:
    from sklearn.base import ClassifierMixin
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
except ImportError:  # pragma: no cover
    ClassifierMixin = None
    ExtraTreesClassifier = None
    GradientBoostingClassifier = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None


class SklearnBinaryClassifier(NonGPBinaryClassificationMixin, nn.Module):
    """Base adapter preserving robotorchan's binary prediction contract."""

    estimator_class: ClassVar[type | None] = None

    def __init__(self, train_X: Tensor, train_Y: Tensor, **estimator_kwargs: Any) -> None:
        super().__init__()
        if self.estimator_class is None or ClassifierMixin is None:
            raise ImportError("This classifier requires scikit-learn.")
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        labels = train_Y.squeeze(-1) if train_Y.ndim == 2 else train_Y
        validate_binary_labels(labels)
        if labels.ndim != 1 or labels.shape[0] != train_X.shape[0]:
            raise ValueError("train_Y must contain one binary label per training row.")
        self.register_buffer("raw_train_X", train_X.clone())
        self.register_buffer("raw_train_Y", labels.clone())
        self._estimator = self.estimator_class(**estimator_kwargs)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying estimator has been fitted."""
        return self._is_fitted

    def fit(self) -> None:
        """Fit the estimator on constructor-level raw training data."""
        X = self.raw_train_X.detach().cpu().numpy()
        y = self.raw_train_Y.detach().cpu().numpy()
        self._estimator.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        """Return fitted class probabilities in class-label order (0, 1)."""
        del kwargs
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        original_shape = X.shape[:-1]
        flat_X = X.detach().cpu().reshape(-1, X.shape[-1]).numpy()
        probabilities = self._estimator.predict_proba(flat_X)
        result = torch.as_tensor(probabilities, dtype=X.dtype, device=X.device)
        return result.reshape(*original_shape, self.num_classes)


class RandomForestBinaryClassifier(SklearnBinaryClassifier):
    """Binary random-forest classifier."""

    estimator_class = RandomForestClassifier


class ExtraTreesBinaryClassifier(SklearnBinaryClassifier):
    """Binary extremely-randomized-trees classifier."""

    estimator_class = ExtraTreesClassifier


class GradientBoostingBinaryClassifier(SklearnBinaryClassifier):
    """Binary gradient-boosting classifier."""

    estimator_class = GradientBoostingClassifier


class HistGradientBoostingBinaryClassifier(SklearnBinaryClassifier):
    """Binary histogram gradient-boosting classifier."""

    estimator_class = HistGradientBoostingClassifier
