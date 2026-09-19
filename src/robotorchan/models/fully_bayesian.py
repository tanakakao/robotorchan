"""Fully Bayesian SAAS wrappers with robotorchan model conventions."""

from __future__ import annotations

from typing import Any

from botorch.models.fully_bayesian import (
    SaasFullyBayesianSingleTaskGP as BoTorchSaasFullyBayesianSingleTaskGP,
)
from botorch.models.fully_bayesian_multitask import MultitaskSaasPyroModel
from botorch.models.fully_bayesian_multitask import (
    SaasFullyBayesianMultiTaskGP as BoTorchSaasFullyBayesianMultiTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
import torch
from torch import Tensor

from robotorchan.models.base import (
    CategoricalOneHotInputTransform,
    ModelTrainingMixin,
    SupervisedTrainingDataMixin,
    normalize_feature_dims,
)


class SaasFullyBayesianSingleTaskGP(
    SupervisedTrainingDataMixin,
    ModelTrainingMixin,
    BoTorchSaasFullyBayesianSingleTaskGP,
):
    """BoTorch SAAS single-task GP with robotorchan raw-data conventions.

    Fully Bayesian fitting remains delegated to BoTorch's
    ``fit_fully_bayesian_model_nuts``. This wrapper only retains the caller's
    untransformed training tensors and makes the lack of MLL-style fitting
    explicit through ``supports_mll = False``.
    """

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


class SaasFullyBayesianMultiTaskGP(
    SupervisedTrainingDataMixin,
    ModelTrainingMixin,
    BoTorchSaasFullyBayesianMultiTaskGP,
):
    """BoTorch SAAS multi-task GP with robotorchan raw-data conventions.

    The long-format task-feature representation, Pyro model, MCMC loading,
    transforms, and posterior computation are inherited from BoTorch. Fitting
    is performed with ``fit_fully_bayesian_model_nuts`` rather than an MLL.
    """

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


class MixedSaasFullyBayesianSingleTaskGP(SaasFullyBayesianSingleTaskGP):
    """Fully Bayesian SAAS GP with model-owned categorical one-hot encoding."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        use_input_warping: bool = False,
        indices_to_warp: list[int] | None = None,
    ) -> None:
        if input_transform is not None:
            raise ValueError(
                "MixedSaasFullyBayesianSingleTaskGP owns its categorical input transform."
            )
        raw_train_X = train_X.detach().clone()
        transform = CategoricalOneHotInputTransform(train_X, cat_dims)
        encoded_train_X = transform(train_X)
        if use_input_warping:
            if indices_to_warp is None:
                indices_to_warp = [
                    transform.encoded_scalar_index(dim)
                    for dim in range(transform.raw_input_dim)
                    if dim not in transform.cat_dims
                ]
            else:
                raw_warp_dims = normalize_feature_dims(
                    indices_to_warp,
                    transform.raw_input_dim,
                    name="indices_to_warp",
                    require_nonempty=False,
                )
                overlap = set(raw_warp_dims).intersection(transform.cat_dims)
                if overlap:
                    raise ValueError(
                        "indices_to_warp must not include categorical features."
                    )
                indices_to_warp = [
                    transform.encoded_scalar_index(dim) for dim in raw_warp_dims
                ]
        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            outcome_transform=outcome_transform,
            use_input_warping=use_input_warping,
            indices_to_warp=indices_to_warp,
        )
        self.mixed_input_transform = transform
        self.cat_dims = transform.cat_dims
        self.raw_input_dim = transform.raw_input_dim
        self.encoded_input_dim = transform.encoded_input_dim
        self._store_supervised_training_data(raw_train_X, train_Y, train_Yvar)

    @property
    def category_values(self) -> tuple[Tensor, ...]:
        """Observed category values in cat_dims order."""
        return self.mixed_input_transform.category_values

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the posterior from raw mixed-space inputs."""
        return super().posterior(self.mixed_input_transform(X), *args, **kwargs)


class MixedSaasFullyBayesianMultiTaskGP(SaasFullyBayesianMultiTaskGP):
    """Fully Bayesian multi-task SAAS GP with internal categorical encoding."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
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
        if input_transform is not None:
            raise ValueError(
                "MixedSaasFullyBayesianMultiTaskGP owns its categorical input transform."
            )
        input_dim = train_X.shape[-1]
        resolved_task_feature = normalize_feature_dims(
            [task_feature], input_dim, name="task_feature"
        )[0]
        transform = CategoricalOneHotInputTransform(
            train_X,
            cat_dims,
            excluded_dims=[resolved_task_feature],
        )
        raw_train_X = train_X.detach().clone()
        encoded_train_X = transform(train_X)
        encoded_task_feature = transform.encoded_scalar_index(resolved_task_feature)
        super().__init__(
            train_X=encoded_train_X,
            train_Y=train_Y,
            task_feature=encoded_task_feature,
            train_Yvar=train_Yvar,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            pyro_model=pyro_model,
            validate_task_values=validate_task_values,
        )
        self.mixed_input_transform = transform
        self.cat_dims = transform.cat_dims
        self.raw_input_dim = input_dim
        self.raw_task_feature = resolved_task_feature
        self.encoded_task_feature = encoded_task_feature
        self.encoded_input_dim = transform.encoded_input_dim
        self._store_supervised_training_data(raw_train_X, train_Y, train_Yvar)

    @property
    def category_values(self) -> tuple[Tensor, ...]:
        """Observed category values in cat_dims order."""
        return self.mixed_input_transform.category_values

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Evaluate posterior from raw design inputs without the task column."""
        design_cat_dims = tuple(
            dim if dim < self.raw_task_feature else dim - 1 for dim in self.cat_dims
        )
        design_train_X = torch.empty(
            (*X.shape[:-1], self.raw_input_dim - 1),
            dtype=X.dtype,
            device=X.device,
        )
        raw_design_dims = [
            dim for dim in range(self.raw_input_dim) if dim != self.raw_task_feature
        ]
        design_train_X.copy_(X)
        category_values = self.category_values
        parts: list[Tensor] = []
        category_by_dim = dict(zip(design_cat_dims, category_values, strict=True))
        for dim in range(design_train_X.shape[-1]):
            if dim not in category_by_dim:
                parts.append(design_train_X[..., dim : dim + 1])
                continue
            values = category_by_dim[dim].to(device=X.device, dtype=X.dtype)
            shape = (1,) * (X.ndim - 1) + (values.numel(),)
            encoded = X[..., dim : dim + 1] == values.reshape(shape)
            if not torch.all(encoded.sum(dim=-1) == 1):
                raise ValueError(
                    f"Input contains an unseen category in categorical feature {dim}."
                )
            parts.append(encoded.to(dtype=X.dtype))
        encoded_X = torch.cat(parts, dim=-1)
        return super().posterior(encoded_X, *args, **kwargs)
