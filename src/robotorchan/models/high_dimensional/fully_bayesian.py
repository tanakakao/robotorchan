"""Fully Bayesian SAAS wrappers with robotorchan model conventions."""

from __future__ import annotations

from typing import Any

import torch
from botorch.fit import fit_fully_bayesian_model_nuts
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

    supports_fit = True
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


