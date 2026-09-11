"""Multi-task GP wrappers with robotorchan model conventions."""

from __future__ import annotations

from typing import Any

from botorch.models import (
    KroneckerMultiTaskGP as BoTorchKroneckerMultiTaskGP,
    MultiTaskGP as BoTorchMultiTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood, MultitaskGaussianLikelihood
from gpytorch.module import Module
from gpytorch.priors import Prior
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class MultiTaskGP(ExactGPModelMixin, BoTorchMultiTaskGP):
    """BoTorch ``MultiTaskGP`` with robotorchan convenience features.

    BoTorch's long-format multi-task representation is preserved unchanged:
    ``train_X`` contains the task feature and ``train_Y`` contains one observed
    outcome per row. robotorchan only retains the caller-supplied raw tensors
    and provides the common exact-GP training interface.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
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
        """Initialize the model while retaining caller-supplied raw tensors."""
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
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=raw_train_Yvar,
        )


class KroneckerMultiTaskGP(ExactGPModelMixin, BoTorchKroneckerMultiTaskGP):
    """BoTorch ``KroneckerMultiTaskGP`` with robotorchan conveniences.

    This preserves BoTorch's block-design representation, where every task is
    observed at every row of ``train_X`` and ``train_Y`` has shape ``n x m``.
    ``raw_train_Yvar`` is retained as ``None`` because the upstream constructor
    does not accept per-observation ``train_Yvar``.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: MultitaskGaussianLikelihood | None = None,
        data_covar_module: Module | None = None,
        task_covar_prior: Prior | None = None,
        rank: int | None = None,
        outcome_transform: OutcomeTransform | None = None,
        input_transform: InputTransform | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model while retaining caller-supplied raw tensors."""
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            data_covar_module=data_covar_module,
            task_covar_prior=task_covar_prior,
            rank=rank,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            **kwargs,
        )
        self._store_supervised_training_data(
            train_X=raw_train_X,
            train_Y=raw_train_Y,
            train_Yvar=None,
        )
