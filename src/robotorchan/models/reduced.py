"""GP wrappers with optional input and output dimensionality reduction."""

from __future__ import annotations

from typing import Any

from botorch.models import SingleTaskGP as BoTorchSingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin
from robotorchan.models.output_reduction import OutputPCAReducer, OutputPLSReducer
from robotorchan.models.reduction import (
    InputReducer,
    OutputReducer,
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
    ReductionMixin,
)


class ReducedGP(ReductionMixin, ExactGPModelMixin, BoTorchSingleTaskGP):
    """Single-task GP with optional frozen input and output reduction.

    Training and candidate inputs are accepted in the original input space.
    When an input reducer is configured, the underlying GP is trained and
    evaluated in the reduced input space. When an output reducer is configured,
    training outcomes are reduced before GP construction and the public
    posterior is reconstructed in the original outcome space.

    ``input_transform`` is applied by the underlying ``SingleTaskGP`` after
    input reduction. ``outcome_transform`` is likewise applied after output
    reduction, so both transforms must be configured for latent dimensions.

    Args:
        train_X: Original training inputs with shape ``n x d``.
        train_Y: Original training outcomes with shape ``n x m``.
        input_reducer: Optional reducer for the input space.
        output_reducer: Optional reducer for the outcome space.
        train_Yvar: Optional observation-noise variances. Output reduction with
            explicit ``train_Yvar`` is not supported because the noise must be
            transformed consistently into latent outcome coordinates.
        likelihood: Optional GPyTorch likelihood.
        covar_module: Optional covariance module for the GP input space.
        mean_module: Optional mean module.
        outcome_transform: Optional BoTorch outcome transform in latent output space.
        input_transform: Optional BoTorch input transform in latent input space.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_reducer: InputReducer | None = None,
        output_reducer: OutputReducer | None = None,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if output_reducer is not None and train_Yvar is not None:
            raise NotImplementedError("Explicit train_Yvar is not supported with output reduction.")

        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()
        self._original_input_dim_value = train_X.shape[-1]
        self._original_output_dim_value = train_Y.shape[-1]

        reduced_train_X = (
            train_X if input_reducer is None else input_reducer.fit_transform(train_X, train_Y)
        )
        reduced_train_Y = (
            train_Y if output_reducer is None else output_reducer.fit_transform(train_Y, train_X)
        )

        super().__init__(
            train_X=reduced_train_X,
            train_Y=reduced_train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._set_reducers(
            input_reducer=input_reducer,
            output_reducer=output_reducer,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )

    @property
    def original_input_dim(self) -> int:
        """Input dimensionality expected by the public model interface."""
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        """Input dimensionality used by the underlying GP."""
        if self.input_reducer is None:
            return self.original_input_dim
        return self.input_reducer.output_dim

    @property
    def original_output_dim(self) -> int:
        """Outcome dimensionality exposed by the public posterior."""
        return self._original_output_dim_value

    @property
    def reduced_output_dim(self) -> int:
        """Outcome dimensionality modeled by the underlying GP."""
        if self.output_reducer is None:
            return self.original_output_dim
        return self.output_reducer.output_dim

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        """Accept original-space or already-reduced inputs."""
        if self.input_reducer is None:
            if X.shape[-1] != self.original_input_dim:
                raise ValueError(
                    f"Expected final input dimension {self.original_input_dim}, got {X.shape[-1]}."
                )
            return X
        if X.shape[-1] == self.original_input_dim:
            return self._transform_inputs(X)
        if X.shape[-1] == self.reduced_input_dim:
            return X
        raise ValueError(
            "Expected final input dimension "
            f"{self.original_input_dim} (original) or {self.reduced_input_dim} (reduced), "
            f"got {X.shape[-1]}."
        )

    def _prepare_outputs(self, Y: Tensor) -> Tensor:
        """Accept original-space or already-reduced outcomes."""
        if self.output_reducer is None:
            if Y.shape[-1] != self.original_output_dim:
                raise ValueError(
                    f"Expected final output dimension {self.original_output_dim}, "
                    f"got {Y.shape[-1]}."
                )
            return Y
        if Y.shape[-1] == self.original_output_dim:
            return self.output_reducer.transform(Y)
        if Y.shape[-1] == self.reduced_output_dim:
            return Y
        raise ValueError(
            "Expected final output dimension "
            f"{self.original_output_dim} (original) or {self.reduced_output_dim} (reduced), "
            f"got {Y.shape[-1]}."
        )

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any | None = None,
    ):
        """Evaluate the posterior from original-space candidate inputs."""
        if self.output_reducer is not None:
            if output_indices is not None:
                raise NotImplementedError(
                    "output_indices is not yet supported with output reduction."
                )
            if posterior_transform is not None:
                raise NotImplementedError(
                    "posterior_transform is not yet supported with output reduction."
                )
            if isinstance(observation_noise, Tensor):
                raise NotImplementedError(
                    "Tensor-valued observation_noise is not supported with output reduction."
                )

        posterior = super().posterior(
            self._prepare_inputs(X),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
        return self._restore_output_posterior(posterior)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        """Condition on observations in the original input and output spaces."""
        if self.output_reducer is not None and kwargs.get("noise") is not None:
            raise NotImplementedError(
                "Explicit observation noise is not supported with output reduction."
            )
        return super().condition_on_observations(
            X=self._prepare_inputs(X),
            Y=self._prepare_outputs(Y),
            **kwargs,
        )


class PCAGP(ReducedGP):
    """Single-task GP using a frozen PCA projection of the input space."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        *,
        center: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=PCAInputReducer(n_components=n_components, center=center),
            **kwargs,
        )


class PLSGP(ReducedGP):
    """Single-task GP using a supervised PLS projection of the input space."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        *,
        center: bool = True,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=PLSInputReducer(
                n_components=n_components,
                center=center,
                eps=eps,
            ),
            **kwargs,
        )


class RandomProjectionGP(ReducedGP):
    """Single-task GP using a frozen Gaussian random input projection."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        *,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=RandomProjectionInputReducer(
                n_components=n_components,
                random_state=random_state,
            ),
            **kwargs,
        )


class OutputPCAGP(ReducedGP):
    """Single-task GP modeling a PCA-compressed high-dimensional output space."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        *,
        center: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            output_reducer=OutputPCAReducer(
                n_components=n_components,
                center=center,
            ),
            **kwargs,
        )


class OutputPLSGP(ReducedGP):
    """Single-task GP modeling a supervised PLS-compressed output space."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        n_components: int,
        *,
        center: bool = True,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            output_reducer=OutputPLSReducer(
                n_components=n_components,
                center=center,
                eps=eps,
            ),
            **kwargs,
        )
