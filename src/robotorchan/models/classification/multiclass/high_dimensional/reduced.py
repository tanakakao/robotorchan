"""Input-reduced multiclass and ordinal GP classifiers.

The reducer is fitted once on the original training data; the latent GPs
operate in the frozen reduced coordinates. Public predictions accept raw X.
"""

from __future__ import annotations

from torch import Tensor

from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.reduction.base import InputReducer
from robotorchan.reduction.input import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)


class _ReducedInputMixin:
    """Map original-space candidates into the fitted latent input space."""

    input_reducer: InputReducer
    _original_input_dim_value: int

    @property
    def original_input_dim(self) -> int:
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        return self.input_reducer.output_dim

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] == self.original_input_dim:
            return self.input_reducer.transform(X)
        if X.shape[-1] == self.reduced_input_dim:
            return X
        raise ValueError(
            f"Expected input dimension {self.original_input_dim} (original) "
            f"or {self.reduced_input_dim} (reduced), got {X.shape[-1]}."
        )

    def posterior(self, X: Tensor, **kwargs: object):
        return super().posterior(self._prepare_inputs(X), **kwargs)


class ReducedMulticlassGPClassifier(_ReducedInputMixin, MulticlassSingleTaskGPClassifier):
    """Multiclass classifier with a frozen, reusable input reducer."""

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
        input_reducer: InputReducer,
    ) -> None:
        reduced = (
            input_reducer.transform(train_X)
            if input_reducer.is_fitted
            else input_reducer.fit_transform(train_X, train_Y.to(dtype=train_X.dtype))
        )
        super().__init__(reduced, train_Y, num_classes=num_classes)
        self.input_reducer = input_reducer
        self._original_input_dim_value = train_X.shape[-1]


class ReducedOrdinalGPClassifier(_ReducedInputMixin, OrdinalSingleTaskGPClassifier):
    """Ordinal-probit classifier with a frozen, reusable input reducer."""

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
        input_reducer: InputReducer,
    ) -> None:
        reduced = (
            input_reducer.transform(train_X)
            if input_reducer.is_fitted
            else input_reducer.fit_transform(train_X, train_Y.to(dtype=train_X.dtype))
        )
        super().__init__(reduced, train_Y, num_classes=num_classes)
        self.input_reducer = input_reducer
        self._original_input_dim_value = train_X.shape[-1]


class PCAMulticlassGPClassifier(ReducedMulticlassGPClassifier):
    def __init__(self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
                 n_components: int, center: bool = True) -> None:
        super().__init__(train_X, train_Y, num_classes=num_classes,
                         input_reducer=PCAInputReducer(
                             n_components=n_components, center=center
                         ))


class PCAOrdinalGPClassifier(ReducedOrdinalGPClassifier):
    def __init__(self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
                 n_components: int, center: bool = True) -> None:
        super().__init__(train_X, train_Y, num_classes=num_classes,
                         input_reducer=PCAInputReducer(
                             n_components=n_components, center=center
                         ))


class PLSMulticlassGPClassifier(ReducedMulticlassGPClassifier):
    def __init__(self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
                 n_components: int, center: bool = True, eps: float = 1e-12) -> None:
        super().__init__(train_X, train_Y, num_classes=num_classes,
                         input_reducer=PLSInputReducer(
                             n_components=n_components, center=center, eps=eps
                         ))


class PLSOrdinalGPClassifier(ReducedOrdinalGPClassifier):
    def __init__(self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
                 n_components: int, center: bool = True, eps: float = 1e-12) -> None:
        super().__init__(train_X, train_Y, num_classes=num_classes,
                         input_reducer=PLSInputReducer(
                             n_components=n_components, center=center, eps=eps
                         ))


class RandomProjectionMulticlassGPClassifier(ReducedMulticlassGPClassifier):
    def __init__(self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
                 n_components: int, random_state: int = 0) -> None:
        super().__init__(train_X, train_Y, num_classes=num_classes,
                         input_reducer=RandomProjectionInputReducer(
                             n_components=n_components, random_state=random_state
                         ))


class RandomProjectionOrdinalGPClassifier(ReducedOrdinalGPClassifier):
    def __init__(self, train_X: Tensor, train_Y: Tensor, *, num_classes: int,
                 n_components: int, random_state: int = 0) -> None:
        super().__init__(train_X, train_Y, num_classes=num_classes,
                         input_reducer=RandomProjectionInputReducer(
                             n_components=n_components, random_state=random_state
                         ))
