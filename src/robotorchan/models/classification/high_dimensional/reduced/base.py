"""Input-reduced binary GP classifiers."""

from __future__ import annotations

from torch import Tensor

from robotorchan.models.classification.standard.binary import BinarySingleTaskGPClassifier
from robotorchan.reduction.base import InputReducer
from robotorchan.reduction.input import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)


class ReducedBinarySingleTaskGPClassifier(BinarySingleTaskGPClassifier):
    """Binary classifier trained in a frozen reduced input space."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_reducer: InputReducer,
    ) -> None:
        """Fit or reuse an input reducer and construct the latent GP."""
        raw_train_X = train_X.detach().clone()
        if input_reducer.is_fitted:
            reduced_train_X = input_reducer.transform(train_X)
        else:
            reduced_train_X = input_reducer.fit_transform(train_X, train_Y)
        original_input_dim = train_X.shape[-1]
        super().__init__(train_X=reduced_train_X, train_Y=train_Y)
        self.input_reducer = input_reducer
        self._original_input_dim_value = original_input_dim
        self._store_raw_tensor("train_X", raw_train_X)

    @property
    def original_input_dim(self) -> int:
        """Input dimensionality accepted by the public interface."""
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        """Input dimensionality modeled by the latent GP."""
        return self.input_reducer.output_dim

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        """Accept original-space or already-reduced candidate inputs."""
        if X.shape[-1] == self.original_input_dim:
            return self.input_reducer.transform(X)
        if X.shape[-1] == self.reduced_input_dim:
            return X
        raise ValueError(
            "Expected final input dimension "
            f"{self.original_input_dim} (original) or {self.reduced_input_dim} (reduced), "
            f"got {X.shape[-1]}."
        )

    def posterior(self, X: Tensor, **kwargs: object):
        """Evaluate the latent posterior from original or reduced inputs."""
        return super().posterior(self._prepare_inputs(X), **kwargs)


class PCABinarySingleTaskGPClassifier(ReducedBinarySingleTaskGPClassifier):
    """Binary GP classifier using a frozen PCA input projection."""

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, n_components: int, *, center: bool = True
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            input_reducer=PCAInputReducer(n_components=n_components, center=center),
        )


class PLSBinarySingleTaskGPClassifier(ReducedBinarySingleTaskGPClassifier):
    """Binary GP classifier using a supervised PLS input projection."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        *,
        center: bool = True,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            input_reducer=PLSInputReducer(n_components=n_components, center=center, eps=eps),
        )


class RandomProjectionBinarySingleTaskGPClassifier(ReducedBinarySingleTaskGPClassifier):
    """Binary GP classifier using a frozen Gaussian random projection."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        *,
        random_state: int = 0,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            input_reducer=RandomProjectionInputReducer(
                n_components=n_components, random_state=random_state
            ),
        )
