"""Long-format task-correlated multiclass and ordinal GP classifiers.

An intrinsic coregionalization kernel shares statistical structure across
integer-identified tasks while preserving each classifier's likelihood.
"""

from __future__ import annotations

import torch
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels import IndexKernel, ProductKernel
from torch import Tensor

from robotorchan.models.base import normalize_feature_dims
from robotorchan.models.classification.multiclass.standard.single_task import (
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    OrdinalSingleTaskGPClassifier,
)


def _task_covar_module(
    train_X: Tensor, task_feature: int, rank: int | None
) -> tuple[ProductKernel, int, int, int]:
    """Reuse the ICM construction contract of binary multitask classification."""
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d] with long-format task IDs.")
    input_dim = train_X.shape[-1]
    resolved = normalize_feature_dims([task_feature], input_dim, name="task_feature")[0]
    values = train_X[:, resolved]
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("task_feature must contain finite task identifiers.")
    rounded = values.round()
    if not torch.equal(values, rounded):
        raise ValueError("task_feature values must be integer task identifiers.")
    task_ids = rounded.to(dtype=torch.long)
    if int(task_ids.min()) < 0:
        raise ValueError("task_feature values must be non-negative.")
    num_tasks = int(task_ids.max()) + 1
    resolved_rank = 1 if rank is None else rank
    if not 1 <= resolved_rank <= num_tasks:
        raise ValueError("rank must be between 1 and the number of indexed tasks.")
    data_dims = [dim for dim in range(input_dim) if dim != resolved]
    if not data_dims:
        raise ValueError("train_X must have at least one non-task feature.")
    data_kernel = get_covar_module_with_dim_scaled_prior(
        ard_num_dims=len(data_dims),
        active_dims=data_dims,
        batch_shape=train_X.shape[:-2],
    )
    task_kernel = IndexKernel(
        num_tasks=num_tasks,
        rank=resolved_rank,
        active_dims=[resolved],
    )
    return ProductKernel(data_kernel, task_kernel), resolved, num_tasks, resolved_rank


class MultiTaskMulticlassGPClassifier(MulticlassSingleTaskGPClassifier):
    """One categorical label per row, with an integer task feature in train_X."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        num_classes: int,
        rank: int | None = None,
    ) -> None:
        kernel, task_dim, num_tasks, resolved_rank = _task_covar_module(
            train_X, task_feature, rank
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.task_feature = task_dim
        self.num_tasks = num_tasks
        self.rank = resolved_rank


class MultiTaskOrdinalGPClassifier(OrdinalSingleTaskGPClassifier):
    """Ordered-probit classification with an ICM kernel over tasks."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        *,
        num_classes: int,
        rank: int | None = None,
    ) -> None:
        kernel, task_dim, num_tasks, resolved_rank = _task_covar_module(
            train_X, task_feature, rank
        )
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.task_feature = task_dim
        self.num_tasks = num_tasks
        self.rank = resolved_rank
