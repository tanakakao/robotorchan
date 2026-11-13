"""Contextual and hierarchical binary variational GP classifiers."""

from __future__ import annotations

import torch
from botorch.models.transforms.input import InputTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import AdditiveKernel, IndexKernel, ProductKernel
from gpytorch.means import Mean
from gpytorch.variational import VariationalStrategy, _VariationalDistribution, _VariationalStrategy
from torch import Tensor

from robotorchan.models.base import normalize_feature_dims
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


def _index_kernel(
    train_X: Tensor, feature: int, rank: int | None
) -> tuple[IndexKernel, int, int]:
    values = torch.unique(train_X[..., feature])
    if not torch.isfinite(values).all() or not torch.equal(values, values.round()):
        raise ValueError("Structural feature values must be finite integer identifiers.")
    ids = values.to(torch.long)
    if ids.numel() == 0 or int(ids.min()) < 0:
        raise ValueError("Structural feature identifiers must be non-negative.")
    count = int(ids.max()) + 1
    resolved_rank = 1 if rank is None else rank
    if not isinstance(resolved_rank, int) or not 1 <= resolved_rank <= count:
        raise ValueError("rank must be between 1 and the number of indexed levels.")
    kernel = IndexKernel(num_tasks=count, rank=resolved_rank, active_dims=[feature])
    return kernel, count, resolved_rank


class ContextualBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Binary GP with shared covariance and context-specific deviations."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        context_feature: int,
        *,
        rank: int | None = None,
        learn_inducing_points: bool = True,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        context = normalize_feature_dims(
            [context_feature], train_X.shape[-1], name="context_feature"
        )[0]
        data_dims = [i for i in range(train_X.shape[-1]) if i != context]
        if not data_dims:
            raise ValueError("At least one non-context feature is required.")
        context_kernel, count, resolved_rank = _index_kernel(train_X, context, rank)

        def data_kernel():
            return get_covar_module_with_dim_scaled_prior(
                ard_num_dims=len(data_dims),
                batch_shape=train_X.shape[:-2],
                active_dims=data_dims,
            )

        covariance = AdditiveKernel(
            data_kernel(),
            ProductKernel(data_kernel(), context_kernel),
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covariance,
            learn_inducing_points=learn_inducing_points,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.context_feature = context
        self.num_contexts = count
        self.rank = resolved_rank


class HierarchicalBinaryGPClassifier(BinarySingleTaskGPClassifier):
    """Binary GP with global, group, and nested-subgroup covariance components."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        group_feature: int,
        subgroup_feature: int,
        *,
        group_rank: int | None = None,
        subgroup_rank: int | None = None,
        learn_inducing_points: bool = True,
        mean_module: Mean | None = None,
        variational_distribution: _VariationalDistribution | None = None,
        variational_strategy: type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Tensor | int | None = None,
        inducing_point_allocator: InducingPointAllocator | None = None,
        input_transform: InputTransform | None = None,
    ) -> None:
        group, subgroup = normalize_feature_dims(
            [group_feature, subgroup_feature], train_X.shape[-1], name="hierarchy_features"
        )
        data_dims = [i for i in range(train_X.shape[-1]) if i not in (group, subgroup)]
        if not data_dims:
            raise ValueError("At least one non-hierarchical feature is required.")
        group_kernel, num_groups, resolved_group_rank = _index_kernel(train_X, group, group_rank)
        subgroup_kernel, num_subgroups, resolved_subgroup_rank = _index_kernel(
            train_X, subgroup, subgroup_rank
        )
        # Subgroup identifiers must be globally unique across parent groups.
        subgroup_ids = train_X[..., subgroup].reshape(-1)
        group_ids = train_X[..., group].reshape(-1)
        for value in torch.unique(subgroup_ids):
            if torch.unique(group_ids[subgroup_ids == value]).numel() != 1:
                raise ValueError("Each subgroup identifier must belong to exactly one group.")

        def data_kernel():
            return get_covar_module_with_dim_scaled_prior(
                ard_num_dims=len(data_dims),
                batch_shape=train_X.shape[:-2],
                active_dims=data_dims,
            )

        covariance = AdditiveKernel(
            data_kernel(),
            ProductKernel(data_kernel(), group_kernel),
            ProductKernel(data_kernel(), subgroup_kernel),
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covariance,
            learn_inducing_points=learn_inducing_points,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=inducing_point_allocator,
            input_transform=input_transform,
        )
        self.group_feature = group
        self.subgroup_feature = subgroup
        self.num_groups = num_groups
        self.num_subgroups = num_subgroups
        self.group_rank = resolved_group_rank
        self.subgroup_rank = resolved_subgroup_rank
