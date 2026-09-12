"""Hierarchical search-space GP wrappers with robotorchan conventions."""

from __future__ import annotations

import torch
from botorch.models.hierarchical.conditional_kernel_gp import (
    LOG_OUTPUTSCALE_CONSTRAINT,
    HierarchicalConditionalKernel,
    HierarchicalConditionalKernelGP as BoTorchHierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP as BoTorchHierarchicalConditionalKernelMultiTaskGP,
    _transform_hierarchical_dependencies,
)
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.map_saas import add_saas_prior
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels import AdditiveKernel, MaternKernel, ProductKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.priors import Prior
from torch import Tensor
from torch.nn import ModuleList

from robotorchan.models.base import ExactGPModelMixin, _normalize_cat_dims, _normalize_dims


class _MixedHierarchicalConditionalKernel(HierarchicalConditionalKernel):
    """Hierarchical kernel whose active blocks support categorical design variables."""

    def __init__(self, *args, cat_dims: list[int], **kwargs) -> None:
        dim = kwargs.get("dim", args[0] if args else None)
        if dim is None:
            raise ValueError("dim is required.")
        self.cat_dims = tuple(_normalize_cat_dims(cat_dims, dim))
        super().__init__(*args, **kwargs)

    def _continuous_kernel(self, dims: list[int]):
        if self.use_saas_prior:
            kernel = MaternKernel(nu=2.5, ard_num_dims=len(dims), active_dims=dims)
            add_saas_prior(kernel)
            return kernel
        kernel = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(dims), use_rbf_kernel=False, active_dims=dims
        )
        kernel.lengthscale = 1.0
        return kernel

    def construct_individual_kernels(self) -> ModuleList:
        modules = ModuleList()
        categorical = set(self.cat_dims)

        for indices in self.partition:
            cat = [i for i in indices if i in categorical]
            cont = [i for i in indices if i not in categorical]

            if cat and cont:
                block_kernel = AdditiveKernel(
                    self._continuous_kernel(cont),
                    CategoricalKernel(ard_num_dims=len(cat), active_dims=cat),
                    ProductKernel(
                        self._continuous_kernel(cont),
                        CategoricalKernel(ard_num_dims=len(cat), active_dims=cat),
                    ),
                )
            elif cat:
                block_kernel = CategoricalKernel(ard_num_dims=len(cat), active_dims=cat)
            else:
                block_kernel = self._continuous_kernel(cont)

            if self.use_outputscale:
                block_kernel = ScaleKernel(
                    block_kernel,
                    outputscale_constraint=LOG_OUTPUTSCALE_CONSTRAINT,
                )
            modules.append(block_kernel)

        return modules


def _parent_dims(hierarchical_dependencies: dict[int, dict[int | float, list[int]]]) -> set[int]:
    return set(hierarchical_dependencies)


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


class MixedHierarchicalConditionalKernelGP(HierarchicalConditionalKernelGP):
    """Hierarchical GP with native categorical covariance inside active blocks."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        cat_dims: list[int],
        eval_hierarchical_features: bool = True,
        separate_hierarchical_features: bool = True,
        train_Yvar: Tensor | None = None,
        use_saas_prior: bool = True,
        use_outputscale: bool = True,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ) -> None:
        normalized = _normalize_cat_dims(cat_dims, train_X.shape[-1])
        overlap = set(normalized) & _parent_dims(hierarchical_dependencies)
        if overlap:
            raise ValueError("Hierarchical parent dimensions are structural and cannot be in cat_dims.")

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
        transformed = _transform_hierarchical_dependencies(
            train_X=train_X,
            hierarchical_dependencies=hierarchical_dependencies,
            input_transform=input_transform,
        )
        self.covar_module = _MixedHierarchicalConditionalKernel(
            dim=train_X.shape[-1],
            hierarchical_dependencies=transformed,
            eval_hierarchical_features=eval_hierarchical_features,
            separate_hierarchical_features=separate_hierarchical_features,
            use_saas_prior=use_saas_prior,
            use_outputscale=use_outputscale,
            cat_dims=normalized,
        ).to(train_X)
        self.cat_dims = tuple(normalized)


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


class MixedHierarchicalConditionalKernelMultiTaskGP(HierarchicalConditionalKernelMultiTaskGP):
    """Hierarchical multi-task GP with native categorical data covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        hierarchical_dependencies: dict[int, dict[int | float, list[int]]],
        cat_dims: list[int],
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
        input_dim = train_X.shape[-1]
        task_dim = _normalize_dims([task_feature], input_dim, name="Task")[0]
        normalized = _normalize_cat_dims(cat_dims, input_dim)
        if task_dim in normalized:
            raise ValueError("task_feature must not be included in cat_dims.")
        data_cat_dims = [dim if dim < task_dim else dim - 1 for dim in normalized]
        overlap = set(data_cat_dims) & _parent_dims(hierarchical_dependencies)
        if overlap:
            raise ValueError("Hierarchical parent dimensions are structural and cannot be in cat_dims.")

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
        transformed = _transform_hierarchical_dependencies(
            train_X=train_X,
            hierarchical_dependencies=hierarchical_dependencies,
            input_transform=input_transform,
        )
        replacement = _MixedHierarchicalConditionalKernel(
            dim=input_dim - 1,
            hierarchical_dependencies=transformed,
            eval_hierarchical_features=eval_hierarchical_features,
            separate_hierarchical_features=separate_hierarchical_features,
            use_saas_prior=use_saas_prior,
            use_outputscale=use_outputscale,
            cat_dims=data_cat_dims,
        ).to(train_X)
        original = self.covar_module.kernels[0]
        replacement.active_dims = original.active_dims
        self.covar_module.kernels[0] = replacement
        self.cat_dims = tuple(normalized)
        self.data_cat_dims = tuple(data_cat_dims)
        self.raw_task_feature = task_dim
