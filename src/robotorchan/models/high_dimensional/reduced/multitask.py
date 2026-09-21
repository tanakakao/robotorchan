"""Shared input-reduction plumbing for multi-task GP models."""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood, MultitaskGaussianLikelihood
from gpytorch.module import Module
from gpytorch.priors import Prior
from torch import Tensor

from robotorchan.models.base import normalize_feature_dims
from robotorchan.models.standard.multitask import KroneckerMultiTaskGP, MultiTaskGP
from robotorchan.reduction.base import InputReducer
from robotorchan.reduction.input import (
    PCAInputReducer,
    PLSInputReducer,
    RandomProjectionInputReducer,
)
from robotorchan.reduction.neural import (
    AutoEncoderInputReducer,
    SupervisedAutoEncoderInputReducer,
    SupervisedVAEInputReducer,
    VAEInputReducer,
)


class ReducedMultiTaskGP(MultiTaskGP):
    """Long-format multi-task GP reducing data features but not task identity."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        input_reducer: InputReducer,
        train_Yvar: Tensor | None = None,
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        task_covar_prior: Prior | _DefaultType | None = DEFAULT,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        validate_task_values: bool = True,
    ) -> None:
        input_dim = train_X.shape[-1]
        resolved_task_feature = normalize_feature_dims(
            [task_feature], input_dim, name="task_feature"
        )[0]
        data_dims = tuple(i for i in range(input_dim) if i != resolved_task_feature)
        data_X = train_X[..., list(data_dims)]
        reduced_data_X = (
            input_reducer.transform(data_X)
            if input_reducer.is_fitted
            else input_reducer.fit_transform(data_X, train_Y)
        )
        reduced_task_feature = reduced_data_X.shape[-1]
        reduced_train_X = torch.cat(
            [reduced_data_X, train_X[..., resolved_task_feature : resolved_task_feature + 1]],
            dim=-1,
        )
        super().__init__(
            train_X=reduced_train_X,
            train_Y=train_Y,
            task_feature=reduced_task_feature,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
        )
        self.input_reducer = input_reducer
        self.original_task_feature = resolved_task_feature
        self.reduced_task_feature = reduced_task_feature
        self.data_dims = data_dims
        self._original_input_dim_value = input_dim
        self._store_supervised_training_data(train_X, train_Y, train_Yvar)

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load reducer state and rebuild reducer-dependent GP training inputs."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        train_X = self._prepare_inputs(self.raw_train_X)
        if hasattr(self, "input_transform"):
            train_X = self.transform_inputs(train_X)
        self.set_train_data(inputs=train_X, targets=self.train_targets, strict=False)
        return result

    @property
    def original_input_dim(self) -> int:
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        return self.input_reducer.output_dim + 1

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] == self.reduced_input_dim:
            return X
        if X.shape[-1] != self.original_input_dim:
            raise ValueError(
                f"Expected final input dimension {self.original_input_dim} "
                f"(original) or {self.reduced_input_dim} (reduced), got {X.shape[-1]}."
            )
        reduced_data_X = self.input_reducer.transform(X[..., list(self.data_dims)])
        task = X[..., self.original_task_feature : self.original_task_feature + 1]
        return torch.cat([reduced_data_X, task], dim=-1)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return super().posterior(self._prepare_inputs(X), *args, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        return super().condition_on_observations(X=self._prepare_inputs(X), Y=Y, **kwargs)


class ReducedKroneckerMultiTaskGP(KroneckerMultiTaskGP):
    """Block-design Kronecker multi-task GP with frozen input reduction."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        input_reducer: InputReducer,
        likelihood: MultitaskGaussianLikelihood | None = None,
        data_covar_module: Module | None = None,
        task_covar_prior: Prior | None = None,
        rank: int | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        **kwargs: Any,
    ) -> None:
        reduced_train_X = (
            input_reducer.transform(train_X)
            if input_reducer.is_fitted
            else input_reducer.fit_transform(train_X, train_Y)
        )
        super().__init__(
            train_X=reduced_train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            data_covar_module=data_covar_module,
            task_covar_prior=task_covar_prior,
            rank=rank,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            **kwargs,
        )
        self.input_reducer = input_reducer
        self._original_input_dim_value = train_X.shape[-1]
        self._store_supervised_training_data(train_X, train_Y, None)

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load reducer state and rebuild reducer-dependent GP training inputs."""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        train_X = self._prepare_inputs(self.raw_train_X)
        if hasattr(self, "input_transform"):
            train_X = self.transform_inputs(train_X)
        self.set_train_data(inputs=train_X, targets=self.train_targets, strict=False)
        return result

    @property
    def original_input_dim(self) -> int:
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        return self.input_reducer.output_dim

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] == self.reduced_input_dim:
            return X
        if X.shape[-1] != self.original_input_dim:
            raise ValueError(
                f"Expected final input dimension {self.original_input_dim} "
                f"(original) or {self.reduced_input_dim} (reduced), got {X.shape[-1]}."
            )
        return self.input_reducer.transform(X)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return super().posterior(self._prepare_inputs(X), *args, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        return super().condition_on_observations(X=self._prepare_inputs(X), Y=Y, **kwargs)


class PCAMultiTaskGP(ReducedMultiTaskGP):
    """Long-format multi-task GP using PCA over non-task input features."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        n_components: int,
        *,
        center: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            input_reducer=PCAInputReducer(n_components=n_components, center=center),
            **kwargs,
        )


class PCAKroneckerMultiTaskGP(ReducedKroneckerMultiTaskGP):
    """Block-design Kronecker multi-task GP using PCA over input features."""

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


class RandomProjectionMultiTaskGP(ReducedMultiTaskGP):
    """Long-format multi-task GP using a frozen random projection."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        n_components: int,
        *,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            input_reducer=RandomProjectionInputReducer(
                n_components=n_components,
                random_state=random_state,
            ),
            **kwargs,
        )


class RandomProjectionKroneckerMultiTaskGP(ReducedKroneckerMultiTaskGP):
    """Block-design Kronecker multi-task GP using a frozen random projection."""

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


class PLSMultiTaskGP(ReducedMultiTaskGP):
    """Long-format multi-task GP using supervised PLS over data features.

    The task feature is excluded from the reducer input. PLS is fitted against
    the row-wise long-format observations, so each task observation contributes
    one supervised sample while task identity remains modeled exclusively by
    the multi-task GP covariance.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        n_components: int,
        *,
        center: bool = True,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            input_reducer=PLSInputReducer(
                n_components=n_components,
                center=center,
                eps=eps,
            ),
            **kwargs,
        )


class PLSKroneckerMultiTaskGP(ReducedKroneckerMultiTaskGP):
    """Block-design Kronecker multi-task GP using multi-output supervised PLS.

    PLS is fitted from the shared design matrix and all task outcomes jointly.
    This lets the latent input representation capture directions associated
    with covariance across the complete task-output matrix.
    """

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


class AutoEncoderMultiTaskGP(ReducedMultiTaskGP):
    """Long-format multi-task GP using a frozen autoencoder representation."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            input_reducer=AutoEncoderInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
            ),
            **kwargs,
        )


class AutoEncoderKroneckerMultiTaskGP(ReducedKroneckerMultiTaskGP):
    """Block-design Kronecker multi-task GP using a frozen autoencoder."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=AutoEncoderInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
            ),
            **kwargs,
        )


class VAEMultiTaskGP(ReducedMultiTaskGP):
    """Long-format multi-task GP using a frozen VAE posterior-mean representation."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            input_reducer=VAEInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                beta=beta,
            ),
            **kwargs,
        )


class VAEKroneckerMultiTaskGP(ReducedKroneckerMultiTaskGP):
    """Block-design Kronecker multi-task GP using a frozen VAE representation."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=VAEInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                beta=beta,
            ),
            **kwargs,
        )


class SupervisedAutoEncoderMultiTaskGP(ReducedMultiTaskGP):
    """Long-format multi-task GP using an outcome-aware frozen autoencoder.

    The task feature is excluded from the reducer input. The auxiliary
    supervised head is trained against row-wise long-format outcomes, while
    task identity remains represented only by the multi-task GP.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            input_reducer=SupervisedAutoEncoderInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                supervised_weight=supervised_weight,
                standardize_y=standardize_y,
            ),
            **kwargs,
        )


class SupervisedAutoEncoderKroneckerMultiTaskGP(ReducedKroneckerMultiTaskGP):
    """Kronecker multi-task GP using a multi-output supervised autoencoder."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=SupervisedAutoEncoderInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                supervised_weight=supervised_weight,
                standardize_y=standardize_y,
            ),
            **kwargs,
        )


class SupervisedVAEMultiTaskGP(ReducedMultiTaskGP):
    """Long-format multi-task GP using an outcome-aware frozen VAE."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        beta: float = 1.0,
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            input_reducer=SupervisedVAEInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                beta=beta,
                supervised_weight=supervised_weight,
                standardize_y=standardize_y,
            ),
            **kwargs,
        )


class SupervisedVAEKroneckerMultiTaskGP(ReducedKroneckerMultiTaskGP):
    """Kronecker multi-task GP using a multi-output supervised frozen VAE."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 32),
        activation: str = "gelu",
        epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int | None = None,
        standardize: bool = True,
        eps: float = 1e-8,
        random_state: int = 0,
        beta: float = 1.0,
        supervised_weight: float = 1.0,
        standardize_y: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            input_reducer=SupervisedVAEInputReducer(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                standardize=standardize,
                eps=eps,
                random_state=random_state,
                beta=beta,
                supervised_weight=supervised_weight,
                standardize_y=standardize_y,
            ),
            **kwargs,
        )
