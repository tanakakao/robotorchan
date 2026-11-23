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
        kernel, task_dim, num_tasks, resolved_rank = _task_covar_module(train_X, task_feature, rank)
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
        kernel, task_dim, num_tasks, resolved_rank = _task_covar_module(train_X, task_feature, rank)
        super().__init__(train_X, train_Y, num_classes=num_classes, covar_module=kernel)
        self.task_feature = task_dim
        self.num_tasks = num_tasks
        self.rank = resolved_rank


class _BlockTaskInputMixin:
    """Expand shared inputs into long-format task observations and candidates."""

    block_input_dim: int
    num_tasks: int

    @staticmethod
    def _make_long_training_data(train_X: Tensor, train_Y: Tensor) -> tuple[Tensor, Tensor]:
        if train_X.ndim != 2 or train_Y.ndim != 2:
            raise ValueError("train_X and train_Y must have shapes [n, d] and [n, m].")
        n, dim = train_X.shape
        if train_Y.shape[0] != n or train_Y.shape[1] < 1:
            raise ValueError("train_Y must have shape [n, num_tasks] with num_tasks >= 1.")
        m = train_Y.shape[1]
        features = train_X[:, None, :].expand(n, m, dim)
        task_ids = torch.arange(m, device=train_X.device, dtype=train_X.dtype)
        task_column = task_ids.view(1, m, 1).expand(n, m, 1)
        long_X = torch.cat((features, task_column), dim=-1).reshape(n * m, dim + 1)
        return long_X, train_Y.reshape(-1)

    def _long_candidates(self, X: Tensor) -> Tensor:
        if X.ndim != 2 or X.shape[1] != self.block_input_dim:
            raise ValueError(
                f"Expected block-design candidate X of shape [q, {self.block_input_dim}]."
            )
        q = X.shape[0]
        features = X[:, None, :].expand(q, self.num_tasks, self.block_input_dim)
        task_ids = torch.arange(self.num_tasks, device=X.device, dtype=X.dtype)
        task_column = task_ids.view(1, self.num_tasks, 1).expand(q, self.num_tasks, 1)
        return torch.cat((features, task_column), dim=-1).reshape(
            q * self.num_tasks, self.block_input_dim + 1
        )

    def _posterior_input(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional tensor.")
        if X.shape[1] == self.block_input_dim:
            return self._long_candidates(X)
        if X.shape[1] == self.block_input_dim + 1:
            return X
        raise ValueError(f"Expected {self.block_input_dim} or {self.block_input_dim + 1} features.")


class KroneckerMultiTaskMulticlassGPClassifier(
    _BlockTaskInputMixin, MultiTaskMulticlassGPClassifier
):
    """Block-design multiclass GP using the variational ICM backend.

    The name mirrors the binary model API; this is not an exact Kronecker GP.
    The latent posterior exposes flattened [q * num_tasks, num_classes] events.
    The probability API returns [q, num_tasks, num_classes].
    """

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, *, num_classes: int, rank: int | None = None
    ) -> None:
        long_X, long_Y = self._make_long_training_data(train_X, train_Y)
        super().__init__(long_X, long_Y, task_feature=-1, num_classes=num_classes, rank=rank)
        self.block_input_dim = train_X.shape[-1]

    def posterior(self, X: Tensor, **kwargs: object):
        return super().posterior(self._posterior_input(X), **kwargs)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        block = X.shape[-1] == self.block_input_dim
        values = super().sample_class_probabilities(
            self._posterior_input(X), sample_shape=sample_shape, **kwargs
        )
        if block:
            return values.reshape(*values.shape[:-2], X.shape[0], self.num_tasks, self.num_classes)
        return values

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        long_X = self._long_candidates(X)
        probabilities = super().predict_proba(long_X, **kwargs)
        return probabilities.reshape(X.shape[0], self.num_tasks, self.num_classes)

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)


class KroneckerMultiTaskOrdinalGPClassifier(_BlockTaskInputMixin, MultiTaskOrdinalGPClassifier):
    """Block-design ordinal GP using variational ICM rather than exact Kronecker algebra."""

    def __init__(
        self, train_X: Tensor, train_Y: Tensor, *, num_classes: int, rank: int | None = None
    ) -> None:
        long_X, long_Y = self._make_long_training_data(train_X, train_Y)
        super().__init__(long_X, long_Y, task_feature=-1, num_classes=num_classes, rank=rank)
        self.block_input_dim = train_X.shape[-1]

    def posterior(self, X: Tensor, **kwargs: object):
        return super().posterior(self._posterior_input(X), **kwargs)

    def sample_class_probabilities(
        self,
        X: Tensor,
        sample_shape: torch.Size | None = None,
        **kwargs: object,
    ) -> Tensor:
        block = X.shape[-1] == self.block_input_dim
        values = super().sample_class_probabilities(
            self._posterior_input(X), sample_shape=sample_shape, **kwargs
        )
        if block:
            return values.reshape(*values.shape[:-2], X.shape[0], self.num_tasks, self.num_classes)
        return values

    def probability_posterior(self, X: Tensor, **kwargs: object):
        """Keep the BoTorch probability posterior's flattened candidate axis.

        Its latent posterior has q * num_tasks input rows, so probability
        moments must use the same axis. For task-shaped means, use
        predict_proba(X), which returns [q, num_tasks, num_classes].
        """
        return super().probability_posterior(self._long_candidates(X), **kwargs)

    def predict_proba(self, X: Tensor, **kwargs: object) -> Tensor:
        if X.ndim == 2 and X.shape[1] == self.block_input_dim + 1:
            return super().predict_proba(X, **kwargs)
        probabilities = super().predict_proba(self._long_candidates(X), **kwargs)
        return probabilities.reshape(X.shape[0], self.num_tasks, self.num_classes)

    def predict_class(self, X: Tensor, **kwargs: object) -> Tensor:
        return self.predict_proba(X, **kwargs).argmax(dim=-1)
