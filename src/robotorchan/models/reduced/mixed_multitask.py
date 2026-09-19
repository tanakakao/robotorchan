"""Mixed-input reduction for multi-task Gaussian-process models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood, MultitaskGaussianLikelihood
from gpytorch.priors import Prior
from torch import Tensor

from robotorchan.models.base import normalize_feature_dims
from robotorchan.models.multitask import MixedKroneckerMultiTaskGP, MixedMultiTaskGP
from robotorchan.models.reduced.mixed import MixedInputReducer
from robotorchan.reduction.base import InputReducer


class MixedReducedMultiTaskGP(MixedMultiTaskGP):
    """Long-format mixed multi-task GP reducing continuous data features only."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        cat_dims: list[int],
        *,
        input_reducer: InputReducer,
        train_Yvar: Tensor | None = None,
        mean_module=None,
        cont_kernel_factory: Callable[[torch.Size, int, list[int]], Kernel] | None = None,
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
        task_dim = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
        cats = normalize_feature_dims(
            cat_dims, input_dim, name="cat_dims", excluded_dims=[task_dim]
        )
        data_dims = tuple(i for i in range(input_dim) if i != task_dim)
        data_cat_dims = [data_dims.index(i) for i in cats]
        data_X = train_X[..., list(data_dims)]
        mixed_reducer = MixedInputReducer(
            input_reducer,
            input_dim=len(data_dims),
            cat_dims=data_cat_dims,
        )
        reduced_data_X = (
            mixed_reducer.transform(data_X)
            if mixed_reducer.is_fitted
            else mixed_reducer.fit_transform(data_X, train_Y)
        )
        reduced_task_feature = reduced_data_X.shape[-1]
        task = train_X[..., task_dim : task_dim + 1].to(reduced_data_X)
        reduced_X = torch.cat((reduced_data_X, task), dim=-1)
        super().__init__(
            reduced_X,
            train_Y,
            task_feature=reduced_task_feature,
            cat_dims=mixed_reducer.cat_dims,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            cont_kernel_factory=cont_kernel_factory,
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
        self._mixed_reducer = mixed_reducer
        self.original_task_feature = task_dim
        self.reduced_task_feature = reduced_task_feature
        self.data_dims = data_dims
        self._original_input_dim_value = input_dim
        self._store_supervised_training_data(train_X, train_Y, train_Yvar)

    @property
    def original_input_dim(self) -> int:
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        return self._mixed_reducer.output_dim + 1

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] == self.reduced_input_dim:
            return X
        if X.shape[-1] != self.original_input_dim:
            raise ValueError(
                f"Expected final input dimension {self.original_input_dim} "
                f"(original) or {self.reduced_input_dim} (reduced), got {X.shape[-1]}."
            )
        data_X = X[..., list(self.data_dims)]
        reduced_data_X = self._mixed_reducer.transform(data_X)
        task = X[..., self.original_task_feature : self.original_task_feature + 1].to(
            reduced_data_X
        )
        return torch.cat((reduced_data_X, task), dim=-1)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return super().posterior(self._prepare_inputs(X), *args, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        return super().condition_on_observations(X=self._prepare_inputs(X), Y=Y, **kwargs)


class MixedReducedKroneckerMultiTaskGP(MixedKroneckerMultiTaskGP):
    """Kronecker mixed multi-task GP reducing continuous input features only."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        *,
        input_reducer: InputReducer,
        likelihood: MultitaskGaussianLikelihood | None = None,
        cont_kernel_factory: Callable[[torch.Size, int, list[int]], Kernel] | None = None,
        task_covar_prior: Prior | None = None,
        rank: int | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        **kwargs: Any,
    ) -> None:
        mixed_reducer = MixedInputReducer(
            input_reducer,
            input_dim=train_X.shape[-1],
            cat_dims=cat_dims,
        )
        reduced_X = (
            mixed_reducer.transform(train_X)
            if mixed_reducer.is_fitted
            else mixed_reducer.fit_transform(train_X, train_Y)
        )
        super().__init__(
            reduced_X,
            train_Y,
            cat_dims=mixed_reducer.cat_dims,
            likelihood=likelihood,
            cont_kernel_factory=cont_kernel_factory,
            task_covar_prior=task_covar_prior,
            rank=rank,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            **kwargs,
        )
        self.input_reducer = input_reducer
        self._mixed_reducer = mixed_reducer
        self._original_input_dim_value = train_X.shape[-1]
        self._store_supervised_training_data(train_X, train_Y, None)

    @property
    def original_input_dim(self) -> int:
        return self._original_input_dim_value

    @property
    def reduced_input_dim(self) -> int:
        return self._mixed_reducer.output_dim

    def _prepare_inputs(self, X: Tensor) -> Tensor:
        if X.shape[-1] == self.reduced_input_dim:
            return X
        if X.shape[-1] != self.original_input_dim:
            raise ValueError(
                f"Expected final input dimension {self.original_input_dim} "
                f"(original) or {self.reduced_input_dim} (reduced), got {X.shape[-1]}."
            )
        return self._mixed_reducer.transform(X)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        return super().posterior(self._prepare_inputs(X), *args, **kwargs)

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any):
        return super().condition_on_observations(X=self._prepare_inputs(X), Y=Y, **kwargs)
