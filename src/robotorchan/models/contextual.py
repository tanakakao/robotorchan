"""Contextual GP wrappers with robotorchan model conventions."""

from __future__ import annotations

import torch
from botorch.models.contextual import LCEAGP as BoTorchLCEAGP
from botorch.models.contextual import SACGP as BoTorchSACGP
from botorch.models.contextual_multioutput import LCEMGP as BoTorchLCEMGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import (
    ExactGPModelMixin,
    _make_mixed_covar_module,
    _normalize_cat_dims,
    _normalize_dims,
)


class SACGP(ExactGPModelMixin, BoTorchSACGP):
    """BoTorch structural-additive contextual GP with robotorchan conveniences."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None,
        decomposition: dict[str, list[int]],
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            decomposition=decomposition,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, raw_train_Yvar)


class LCEAGP(ExactGPModelMixin, BoTorchLCEAGP):
    """BoTorch latent-context-embedding additive GP with raw-data retention."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None,
        decomposition: dict[str, list[int]],
        train_embedding: bool = True,
        cat_feature_dict: dict | None = None,
        embs_feature_dict: dict | None = None,
        embs_dim_list: list[int] | None = None,
        context_weight_dict: dict | None = None,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            decomposition=decomposition,
            train_embedding=train_embedding,
            cat_feature_dict=cat_feature_dict,
            embs_feature_dict=embs_feature_dict,
            embs_dim_list=embs_dim_list,
            context_weight_dict=context_weight_dict,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, raw_train_Yvar)


class LCEMGP(ExactGPModelMixin, BoTorchLCEMGP):
    """BoTorch latent-context-embedding multi-output GP with common raw-data API."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        context_cat_feature: Tensor | None = None,
        context_emb_feature: Tensor | None = None,
        embs_dim_list: list[int] | None = None,
        output_tasks: list[int] | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            context_cat_feature=context_cat_feature,
            context_emb_feature=context_emb_feature,
            embs_dim_list=embs_dim_list,
            output_tasks=output_tasks,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, raw_train_Yvar)


class MixedLCEMGP(LCEMGP):
    """LCE-M GP with native mixed covariance on non-task design features.

    ``context_cat_feature`` is context metadata for the latent embedding and is
    deliberately distinct from ``cat_dims``. ``cat_dims`` refers only to
    categorical columns in caller-supplied ``train_X``. The task feature is
    structural and cannot be included in ``cat_dims``.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        cat_dims: list[int],
        train_Yvar: Tensor | None = None,
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        context_cat_feature: Tensor | None = None,
        context_emb_feature: Tensor | None = None,
        embs_dim_list: list[int] | None = None,
        output_tasks: list[int] | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        if covar_module is not None:
            raise ValueError(
                "MixedLCEMGP constructs covar_module from cat_dims; a custom "
                "covar_module is not currently supported."
            )

        input_dim = train_X.shape[-1]
        task_dim = _normalize_dims([task_feature], input_dim, name="Task")[0]
        normalized_cat_dims = _normalize_cat_dims(cat_dims, input_dim)
        if task_dim in normalized_cat_dims:
            raise ValueError("task_feature must not be included in cat_dims.")

        data_cat_dims = [dim if dim < task_dim else dim - 1 for dim in normalized_cat_dims]
        mixed_covar = _make_mixed_covar_module(
            input_dim=input_dim - 1,
            cat_dims=data_cat_dims,
            batch_shape=torch.Size(),
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            train_Yvar=train_Yvar,
            mean_module=mean_module,
            covar_module=mixed_covar,
            likelihood=likelihood,
            context_cat_feature=context_cat_feature,
            context_emb_feature=context_emb_feature,
            embs_dim_list=embs_dim_list,
            output_tasks=output_tasks,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.cat_dims = tuple(normalized_cat_dims)
        self.data_cat_dims = tuple(data_cat_dims)
        self.raw_task_feature = task_dim
