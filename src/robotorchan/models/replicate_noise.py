"""Exact GP using empirical observation noise estimated from replicates."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.models.single_task import SingleTaskGP


class ReplicateNoiseSingleTaskGP(SingleTaskGP):
    """Single-task GP trained on replicate means with empirical mean variance."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        *,
        raw_replicate_X: Tensor,
        raw_replicate_Y: Tensor,
        replicate_counts: Tensor,
        replicate_variance: Tensor,
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self._store_raw_tensor("replicate_X", raw_replicate_X.detach().clone())
        self._store_raw_tensor("replicate_Y", raw_replicate_Y.detach().clone())
        self._store_raw_tensor("replicate_counts", replicate_counts.detach().clone())
        self._store_raw_tensor("replicate_variance", replicate_variance.detach().clone())

    @classmethod
    def from_replicates(
        cls,
        train_X: Tensor,
        train_Y: Tensor,
        *,
        noise_floor: float = 1e-6,
    ) -> ReplicateNoiseSingleTaskGP:
        """Aggregate exact duplicate design rows and estimate variance of each mean."""
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape n x d.")
        if train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("train_Y must have shape n x 1.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError("train_X and train_Y must contain the same number of rows.")
        if not torch.isfinite(train_X).all() or not torch.isfinite(train_Y).all():
            raise ValueError("Replicate training data must be finite.")
        if noise_floor <= 0:
            raise ValueError("noise_floor must be positive.")

        unique_X, inverse, counts = torch.unique(
            train_X,
            dim=0,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        if torch.any(counts < 2):
            raise ValueError("Every design condition must contain at least two replicates.")

        means = []
        variances = []
        for group_idx in range(unique_X.shape[0]):
            values = train_Y[inverse == group_idx]
            means.append(values.mean(dim=0))
            variances.append(values.var(dim=0, unbiased=True))

        group_mean = torch.stack(means)
        replicate_variance = torch.stack(variances)
        count_values = counts.to(dtype=train_Y.dtype).unsqueeze(-1)
        mean_variance = (replicate_variance / count_values).clamp_min(noise_floor)

        return cls(
            train_X=unique_X,
            train_Y=group_mean,
            train_Yvar=mean_variance,
            raw_replicate_X=train_X,
            raw_replicate_Y=train_Y,
            replicate_counts=counts,
            replicate_variance=replicate_variance,
        )

    @property
    def raw_replicate_X(self) -> Tensor:
        """Unaggregated replicate design rows."""
        value = self._get_raw_tensor("replicate_X")
        if value is None:
            raise RuntimeError("raw_replicate_X was unexpectedly stored as None.")
        return value

    @property
    def raw_replicate_Y(self) -> Tensor:
        """Unaggregated replicate observations."""
        value = self._get_raw_tensor("replicate_Y")
        if value is None:
            raise RuntimeError("raw_replicate_Y was unexpectedly stored as None.")
        return value

    @property
    def replicate_counts(self) -> Tensor:
        """Number of observations contributing to each aggregated design row."""
        value = self._get_raw_tensor("replicate_counts")
        if value is None:
            raise RuntimeError("replicate_counts was unexpectedly stored as None.")
        return value

    @property
    def replicate_variance(self) -> Tensor:
        """Unbiased empirical observation variance within each replicate group."""
        value = self._get_raw_tensor("replicate_variance")
        if value is None:
            raise RuntimeError("replicate_variance was unexpectedly stored as None.")
        return value
