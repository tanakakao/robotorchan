"""Hierarchical search-space GP wrappers with robotorchan conventions."""

from __future__ import annotations

from botorch.models.hierarchical.conditional_kernel_gp import (
    HierarchicalConditionalKernelGP as BoTorchHierarchicalConditionalKernelGP,
)
from botorch.models.hierarchical.conditional_kernel_gp import (
    HierarchicalConditionalKernelMultiTaskGP as BoTorchHierarchicalConditionalKernelMultiTaskGP,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.likelihoods import Likelihood
from gpytorch.priors import Prior
from torch import Tensor

from robotorchan.models.base import ExactGPModelMixin


class HierarchicalConditionalKernelGP(
    ExactGPModelMixin,
    BoTorchHierarchicalConditionalKernelGP,
):
    """BoTorch hierarchical single-task GP with robotorchan conveniences."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        train_Yvar: Tensor | None = None,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            hierarchical_dependencies=hierarchical_dependencies,
            eval_hierarchical_features=eval_hierarchical_features,
            separate_hierarchical_features=separate_hierarchical_features,
            train_Yvar=train_Yvar,
            use_saas_prior=use_saas_prior,
            use_outputscale=use_outputscale,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, raw_train_Yvar)


class HierarchicalConditionalKernelMultiTaskGP(
    ExactGPModelMixin,
    BoTorchHierarchicalConditionalKernelMultiTaskGP,
):
    """BoTorch hierarchical multi-task GP with robotorchan conveniences."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        train_Yvar: Tensor | None = None,
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        likelihood: Likelihood | None = None,
        task_covar_prior: Prior | None = None,
        output_tasks: list[int] | None = None,
        rank: int | None = None,
        all_tasks: list[int] | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
        validate_task_values: bool = True,
    ) -> None:
        raw_train_X = train_X.detach().clone()
        raw_train_Y = train_Y.detach().clone()
        raw_train_Yvar = None if train_Yvar is None else train_Yvar.detach().clone()

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            task_feature=task_feature,
            hierarchical_dependencies=hierarchical_dependencies,
            train_Yvar=train_Yvar,
            eval_hierarchical_features=eval_hierarchical_features,
            separate_hierarchical_features=separate_hierarchical_features,
            use_saas_prior=use_saas_prior,
            use_outputscale=use_outputscale,
            likelihood=likelihood,
            task_covar_prior=task_covar_prior,
            output_tasks=output_tasks,
            rank=rank,
            all_tasks=all_tasks,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            validate_task_values=validate_task_values,
        )
        self._store_supervised_training_data(raw_train_X, raw_train_Y, raw_train_Yvar)
