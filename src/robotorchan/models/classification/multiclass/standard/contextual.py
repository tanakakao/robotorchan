"""Contextual and nested-hierarchical multiclass and ordinal GP classifiers.

Reuse the existing binary classifier's additive shared/deviation covariance
design and the existing multiclass and ordered-probit variational inference.
"""

from __future__ import annotations

import torch
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels import AdditiveKernel, IndexKernel, ProductKernel
from torch import Tensor

from robotorchan.models.base import normalize_feature_dims
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)


def _structural_kernel(
    train_X: Tensor,
    feature: int,
    rank: int | None,
) -> tuple[IndexKernel, int, int]:
    """Validate a structural ID column and make its GPyTorch index kernel."""
    ids = train_X[:, feature]
    if ids.numel() == 0 or not torch.isfinite(ids).all():
        raise ValueError("Structural identifiers must be finite non-negative integers.")
    if not torch.equal(ids, ids.round()) or torch.any(ids < 0):
        raise ValueError("Structural identifiers must be finite non-negative integers.")
    count = int(ids.max().item()) + 1
    resolved_rank = 1 if rank is None else rank
    if not isinstance(resolved_rank, int) or not 1 <= resolved_rank <= count:
        raise ValueError("rank must be between 1 and the number of indexed levels.")
    return (
        IndexKernel(num_tasks=count, rank=resolved_rank, active_dims=[feature]),
        count,
        resolved_rank,
    )


def _data_kernel(train_X: Tensor, dimensions: list[int]):
    return get_covar_module_with_dim_scaled_prior(
        ard_num_dims=len(dimensions),
        batch_shape=train_X.shape[:-2],
        active_dims=dimensions,
    )


def _contextual_kernel(
    train_X: Tensor,
    context_feature: int,
    rank: int | None,
) -> tuple[AdditiveKernel, int, int, int]:
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    context = normalize_feature_dims(
        [context_feature], train_X.shape[-1], name="context_feature"
    )[0]
    dimensions = [dim for dim in range(train_X.shape[-1]) if dim != context]
    if not dimensions:
        raise ValueError("At least one non-context feature is required.")
    index_kernel, count, resolved_rank = _structural_kernel(train_X, context, rank)
    covariance = AdditiveKernel(
        _data_kernel(train_X, dimensions),
        ProductKernel(_data_kernel(train_X, dimensions), index_kernel),
    )
    return covariance, context, count, resolved_rank


def _hierarchical_kernel(
    train_X: Tensor,
    group_feature: int,
    subgroup_feature: int,
    group_rank: int | None,
    subgroup_rank: int | None,
) -> tuple[AdditiveKernel, int, int, int, int, int, int]:
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    group, subgroup = normalize_feature_dims(
        [group_feature, subgroup_feature],
        train_X.shape[-1],
        name="hierarchy_features",
    )
    if group == subgroup:
        raise ValueError("group_feature and subgroup_feature must be distinct.")
    dimensions = [dim for dim in range(train_X.shape[-1]) if dim not in (group, subgroup)]
    if not dimensions:
        raise ValueError("At least one non-hierarchical feature is required.")
    group_kernel, num_groups, resolved_group_rank = _structural_kernel(
        train_X, group, group_rank
    )
    subgroup_kernel, num_subgroups, resolved_subgroup_rank = _structural_kernel(
        train_X, subgroup, subgroup_rank
    )
    group_ids = train_X[:, group]
    subgroup_ids = train_X[:, subgroup]
    for value in torch.unique(subgroup_ids):
        if torch.unique(group_ids[subgroup_ids == value]).numel() != 1:
            raise ValueError("Each subgroup identifier must belong to exactly one group.")
    covariance = AdditiveKernel(
        _data_kernel(train_X, dimensions),
        ProductKernel(_data_kernel(train_X, dimensions), group_kernel),
        ProductKernel(_data_kernel(train_X, dimensions), subgroup_kernel),
    )
    return (
        covariance,
        group,
        subgroup,
        num_groups,
        num_subgroups,
        resolved_group_rank,
        resolved_subgroup_rank,
    )


class ContextualMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass categorical GP with shared and context-specific covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        context_feature: int,
        *,
        num_classes: int,
        rank: int | None = None,
    ) -> None:
        kernel, context, count, resolved_rank = _contextual_kernel(
            train_X, context_feature, rank
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.context_feature = context
        self.num_contexts = count
        self.rank = resolved_rank


class ContextualOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordered-probit GP with shared and context-specific covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        context_feature: int,
        *,
        num_classes: int,
        rank: int | None = None,
    ) -> None:
        kernel, context, count, resolved_rank = _contextual_kernel(
            train_X, context_feature, rank
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.context_feature = context
        self.num_contexts = count
        self.rank = resolved_rank


class HierarchicalMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """Multiclass GP with global, group, and nested-subgroup covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        group_feature: int,
        subgroup_feature: int,
        *,
        num_classes: int,
        group_rank: int | None = None,
        subgroup_rank: int | None = None,
    ) -> None:
        result = _hierarchical_kernel(
            train_X, group_feature, subgroup_feature, group_rank, subgroup_rank
        )
        kernel, group, subgroup, n_groups, n_subgroups, g_rank, s_rank = result
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.group_feature = group
        self.subgroup_feature = subgroup
        self.num_groups = n_groups
        self.num_subgroups = n_subgroups
        self.group_rank = g_rank
        self.subgroup_rank = s_rank


class HierarchicalOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordered-probit GP with global, group, and nested-subgroup covariance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        group_feature: int,
        subgroup_feature: int,
        *,
        num_classes: int,
        group_rank: int | None = None,
        subgroup_rank: int | None = None,
    ) -> None:
        result = _hierarchical_kernel(
            train_X, group_feature, subgroup_feature, group_rank, subgroup_rank
        )
        kernel, group, subgroup, n_groups, n_subgroups, g_rank, s_rank = result
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.group_feature = group
        self.subgroup_feature = subgroup
        self.num_groups = n_groups
        self.num_subgroups = n_subgroups
        self.group_rank = g_rank
        self.subgroup_rank = s_rank
