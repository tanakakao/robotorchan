"""GP wrappers that apply frozen dimensionality reduction to model inputs."""

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
from robotorchan.models.reduction import (
    InputReducer,
    OutputReducer,
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
    ReductionMixin,
)


class ReducedGP(ReductionMixin, ExactGPModelMixin, BoTorchSingleTaskGP):
    """Single-task GP evaluated through a frozen input reducer.

    Callers interact with the model in the original input space. Training inputs
    are reduced once during construction, while ``posterior`` and
    ``condition_on_observations`` transparently project later candidate points.
    This keeps acquisition functions, q-batches, input perturbations, and
    fantasy-based workflows on the usual BoTorch model interface.

    ``input_transform`` is applied by the underlying ``SingleTaskGP`` after
    dimensionality reduction, so transforms such as ``Normalize`` must be
    configured for the reduced dimension.

    Args:
        train_X: Original training inputs with shape ``n x d``.
        train_Y: Training outcomes.
        input_reducer: Fitted internally from ``train_X`` and ``train_Y``.
        train_Yvar: Optional observation-noise variances.
        likelihood: Optional GPyTorch likelihood.
        covar_module: Optional covariance module for the latent input space.
        mean_module: Optional mean module.
        outcome_transform: Optional BoTorch outcome transform.
        input_transform: Optional BoTorch transform applied in reduced space.
        output_reducer: Reserved for output-side reduction. Phase 3 supports
            input reduction only and therefore requires ``None``.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_reducer: InputReducer,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        covar_module: Module | None = None,
        mean_module: Mean | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        output_reducer: OutputReducer | None = None,
    ) -> None:
        if output_reducer is not None:
            raise NotImplementedError("Output reduction is not supported until Phase 4.")

        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()
        reduced_train_X = input_reducer.fit_transform(train_X, train_Y)

        super().__init__(
            train_X=reduced_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._set_reducers(input_reducer=input_reducer, output_reducer=None)
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )

    @property
    def original_input_dim(self) -> int:
        """Input dimensionality expected by the public model interface."""
        return self.input_reducer.input_dim

    @property
    def reduced_input_dim(self) -> int:
        """Latent dimensionality used by the underlying GP."""
        return self.input_reducer.output_dim

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        """Accept original-space or already-reduced inputs.

        Supporting already-reduced inputs is important for BoTorch internals
        that may call model methods while conditioning fantasy models.
        """
        if X.shape[-1] == self.original_input_dim:
            return self._transform_inputs(X)
        if X.shape[-1] == self.reduced_input_dim:
            return X
        raise ValueError(
            "Expected final input dimension "
            f"{self.original_input_dim} (original) or {self.reduced_input_dim} (reduced), "
            f"got {X.shape[-1]}."
        )

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any | None = None,
    ):
        """Evaluate the posterior from original-space candidate inputs."""
        posterior = super().posterior(
            self._prepare_inputs(X),
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )
        return self._restore_output_posterior(posterior)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        """Condition on observations supplied in the original input space."""
        return super().condition_on_observations(
            X=self._prepare_inputs(X),
            Y=Y,
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
