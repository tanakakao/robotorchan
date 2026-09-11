"""Fully Bayesian SAAS wrappers with robotorchan model conventions."""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.fully_bayesian import (
    SaasFullyBayesianSingleTaskGP as BoTorchSaasFullyBayesianSingleTaskGP,
)
from botorch.models.fully_bayesian_multitask import MultitaskSaasPyroModel
from botorch.models.fully_bayesian_multitask import (
    SaasFullyBayesianMultiTaskGP as BoTorchSaasFullyBayesianMultiTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from torch import Tensor

from robotorchan.models.base import (
    ModelTrainingMixin,
    SupervisedTrainingDataMixin,
    _normalize_cat_dims,
)


def _collect_category_values(train_X: Tensor, cat_dims: list[int]) -> tuple[Tensor, ...]:
    """Collect sorted category values for each categorical input column."""
    return tuple(torch.unique(train_X[..., dim]).sort().values for dim in cat_dims)


def _one_hot_encode(
    X: Tensor,
    *,
    input_dim: int,
    cat_dims: tuple[int, ...],
    category_values: tuple[Tensor, ...],
) -> Tensor:
    """Encode categorical columns while preserving all other columns in order."""
    if X.shape[-1] != input_dim:
        raise ValueError(
            f"Expected inputs with {input_dim} features, got {X.shape[-1]}."
        )

    value_by_dim = dict(zip(cat_dims, category_values, strict=True))
    parts: list[Tensor] = []
    for dim in range(input_dim):
        if dim not in value_by_dim:
            parts.append(X[..., dim : dim + 1])
            continue

        values = value_by_dim[dim].to(device=X.device, dtype=X.dtype)
        shape = (1,) * (X.ndim - 1) + (values.numel(),)
        encoded = X[..., dim : dim + 1] == values.reshape(shape)
        if not torch.all(encoded.sum(dim=-1) == 1):
            raise ValueError(
                f"Input contains an unseen category in categorical feature {dim}."
            )
        parts.append(encoded.to(dtype=X.dtype))

    return torch.cat(parts, dim=-1)


def _encoded_scalar_index(
    raw_dim: int,
    *,
    input_dim: int,
    cat_dims: tuple[int, ...],
    category_values: tuple[Tensor, ...],
) -> int:
    """Map a non-categorical raw feature index to its encoded scalar index."""
    offset = 0
    value_by_dim = dict(zip(cat_dims, category_values, strict=True))
    for dim in range(input_dim):
        if dim == raw_dim:
            if dim in value_by_dim:
                raise ValueError("Categorical features do not map to one scalar index.")
            return offset
        offset += value_by_dim[dim].numel() if dim in value_by_dim else 1
    raise ValueError(f"Feature index {raw_dim} is out of range.")


class SaasFullyBayesianSingleTaskGP(
    SupervisedTrainingDataMixin,
    ModelTrainingMixin,
    BoTorchSaasFullyBayesianSingleTaskGP,
):
    """BoTorch SAAS single-task GP with robotorchan raw-data conventions."""

    supports_mll = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
    ) -> None:
        """Initialize the SAAS model while retaining caller raw tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            use_input_warping=use_input_warping,
            indices_to_warp=indices_to_warp,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )


class MixedSaasFullyBayesianSingleTaskGP(SaasFullyBayesianSingleTaskGP):
    """Fully Bayesian SAAS GP with internal one-hot categorical encoding.

    The public input space remains the original mixed space. Categorical columns
    listed in ``cat_dims`` are one-hot encoded before the encoded tensor reaches
    BoTorch's JAX/NumPyro NUTS path, and the same learned encoding is applied to
    posterior inputs automatically.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        input_dim = train_X.shape[-1]
        normalized_cat_dims = _normalize_cat_dims(cat_dims, input_dim)
        category_values = _collect_category_values(train_X, normalized_cat_dims)
        encoded_train_X = _one_hot_encode(
            train_X,
            input_dim=input_dim,
            cat_dims=tuple(normalized_cat_dims),
            category_values=category_values,
        )

        if use_input_warping:
            if indices_to_warp is None:
                indices_to_warp = [
                    _encoded_scalar_index(
                        dim,
                        input_dim=input_dim,
                        cat_dims=tuple(normalized_cat_dims),
                        category_values=category_values,
                    )
                    for dim in range(input_dim)
                    if dim not in normalized_cat_dims
                ]
            elif set(indices_to_warp).intersection(normalized_cat_dims):
                raise ValueError("indices_to_warp must not include categorical features.")
            else:
                indices_to_warp = [
                    _encoded_scalar_index(
                        dim,
                        input_dim=input_dim,
                        cat_dims=tuple(normalized_cat_dims),
                        category_values=category_values,
                    )
                    for dim in indices_to_warp
                ]

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            use_input_warping=use_input_warping,
            indices_to_warp=indices_to_warp,
        )

        self.cat_dims = tuple(normalized_cat_dims)
        self.raw_input_dim = input_dim
        self.encoded_input_dim = encoded_train_X.shape[-1]
        self._category_buffer_names: tuple[str, ...] = tuple(
            f"_category_values_{index}" for index in range(len(category_values))
        )
        for name, values in zip(
            self._category_buffer_names, category_values, strict=True
        ):
            self.register_buffer(name, values.detach().clone())
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
        )

    @property
    def category_values(self) -> tuple[Tensor, ...]:
        """Return model-owned category values in ``cat_dims`` order."""
        return tuple(getattr(self, name) for name in self._category_buffer_names)

    def _encode_inputs(self, X: Tensor) -> Tensor:
        return _one_hot_encode(
            X,
            input_dim=self.raw_input_dim,
            cat_dims=self.cat_dims,
            category_values=self.category_values,
        )

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the posterior from inputs in the original mixed space."""
        return super().posterior(self._encode_inputs(X), *args, **kwargs)


class SaasFullyBayesianMultiTaskGP(
    SupervisedTrainingDataMixin,
    ModelTrainingMixin,
    BoTorchSaasFullyBayesianMultiTaskGP,
):
    """BoTorch SAAS multi-task GP with robotorchan raw-data conventions."""

    supports_mll = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        pyro_model: MultitaskSaasPyroModel | None = None,
        validate_task_values: bool = True,
    ) -> None:
        """Initialize the multi-task SAAS model and retain raw training data."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            pyro_model=pyro_model,
            validate_task_values=validate_task_values,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )


class MixedSaasFullyBayesianMultiTaskGP(SaasFullyBayesianMultiTaskGP):
    """Fully Bayesian multi-task SAAS GP with internal one-hot encoding."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        pyro_model: MultitaskSaasPyroModel | None = None,
        validate_task_values: bool = True,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        input_dim = train_X.shape[-1]
        normalized_cat_dims = _normalize_cat_dims(cat_dims, input_dim)
        resolved_task_feature = task_feature + input_dim if task_feature < 0 else task_feature
        if resolved_task_feature < 0 or resolved_task_feature >= input_dim:
            raise ValueError(
                f"task_feature {task_feature} is out of range for input_dim={input_dim}."
            )
        if resolved_task_feature in normalized_cat_dims:
            raise ValueError("task_feature must not also be listed in cat_dims.")

        category_values = _collect_category_values(train_X, normalized_cat_dims)
        encoded_train_X = _one_hot_encode(
            train_X,
            input_dim=input_dim,
            cat_dims=tuple(normalized_cat_dims),
            category_values=category_values,
        )
        encoded_task_feature = _encoded_scalar_index(
            resolved_task_feature,
            input_dim=input_dim,
            cat_dims=tuple(normalized_cat_dims),
            category_values=category_values,
        )

        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            task_feature=encoded_task_feature,
            train_Yvar=train_Yvar,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            pyro_model=pyro_model,
            validate_task_values=validate_task_values,
        )

        self.cat_dims = tuple(normalized_cat_dims)
        self.raw_input_dim = input_dim
        self.raw_task_feature = resolved_task_feature
        self.encoded_task_feature = encoded_task_feature
        self.encoded_input_dim = encoded_train_X.shape[-1]
        self._category_buffer_names: tuple[str, ...] = tuple(
            f"_category_values_{index}" for index in range(len(category_values))
        )
        for name, values in zip(
            self._category_buffer_names, category_values, strict=True
        ):
            self.register_buffer(name, values.detach().clone())
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
        )

    @property
    def category_values(self) -> tuple[Tensor, ...]:
        """Return model-owned category values in ``cat_dims`` order."""
        return tuple(getattr(self, name) for name in self._category_buffer_names)

    def _encode_design_inputs(self, X: Tensor) -> Tensor:
        design_input_dim = self.raw_input_dim - 1
        design_cat_dims = tuple(
            dim if dim < self.raw_task_feature else dim - 1 for dim in self.cat_dims
        )
        return _one_hot_encode(
            X,
            input_dim=design_input_dim,
            cat_dims=design_cat_dims,
            category_values=self.category_values,
        )

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Evaluate posterior from design inputs in the original mixed space."""
        return super().posterior(self._encode_design_inputs(X), *args, **kwargs)
