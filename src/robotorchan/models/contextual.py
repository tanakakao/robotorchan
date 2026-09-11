"""Contextual GP wrappers with robotorchan model conventions."""

from __future__ import annotations

from botorch.models.contextual import LCEAGP as BoTorchLCEAGP
from botorch.models.contextual import SACGP as BoTorchSACGP
from botorch.models.contextual_multioutput import LCEMGP as BoTorchLCEMGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.module import Module
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


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
