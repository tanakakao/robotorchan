"""Binary classification that preserves repeated labels at identical inputs."""

from __future__ import annotations

import torch
from torch import Tensor

from robotorchan.models.classification.binary.robust.base import (
    ClassificationRobustnessType,
    RobustBinaryClassificationMixin,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


def _replicate_label_statistics(
    train_X: Tensor,
    train_Y: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Summarize repeated binary labels without aggregating training targets."""
    if train_X.ndim != 2:
        raise ValueError("train_X must have shape n x d.")
    labels = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[-1] == 1 else train_Y
    if labels.ndim != 1:
        raise ValueError("train_Y must have shape n or n x 1.")
    if train_X.shape[0] != labels.shape[0]:
        raise ValueError("train_X and train_Y must contain the same number of rows.")
    if not torch.isfinite(train_X).all():
        raise ValueError("Replicate training inputs must be finite.")

    unique_X, inverse, counts = torch.unique(
        train_X,
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    if torch.any(counts < 2):
        raise ValueError("Every design condition must contain at least two replicate labels.")

    positive_counts = torch.zeros(
        unique_X.shape[0],
        dtype=labels.dtype,
        device=labels.device,
    )
    positive_counts.scatter_add_(0, inverse, labels)
    count_values = counts.to(dtype=labels.dtype, device=labels.device)
    positive_rate = positive_counts / count_values
    disagreement = 2.0 * positive_rate * (1.0 - positive_rate)
    return unique_X, counts, positive_counts, disagreement


class ReplicateLabelBinarySingleTaskGPClassifier(
    RobustBinaryClassificationMixin,
    BinarySingleTaskGPClassifier,
):
    """Binary GP classifier that retains all labels from replicated conditions."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        **kwargs: object,
    ) -> None:
        unique_X, counts, positive_counts, disagreement = _replicate_label_statistics(
            train_X,
            train_Y,
        )
        super().__init__(train_X=train_X, train_Y=train_Y, **kwargs)
        self._store_raw_tensor("replicate_unique_X", unique_X.detach().clone())
        self._store_raw_tensor("replicate_counts", counts.detach().clone())
        self._store_raw_tensor("replicate_positive_counts", positive_counts.detach().clone())
        self._store_raw_tensor("replicate_disagreement", disagreement.detach().clone())

    @classmethod
    def from_replicates(
        cls,
        train_X: Tensor,
        train_Y: Tensor,
        **kwargs: object,
    ) -> ReplicateLabelBinarySingleTaskGPClassifier:
        """Construct from repeated observations while preserving every binary label."""
        return cls(train_X=train_X, train_Y=train_Y, **kwargs)

    @property
    def replicate_unique_X(self) -> Tensor:
        """Return unique replicated design conditions."""
        value = self._get_raw_tensor("replicate_unique_X")
        if value is None:
            raise RuntimeError("replicate_unique_X was unexpectedly stored as None.")
        return value

    @property
    def replicate_counts(self) -> Tensor:
        """Return observation counts for each replicated condition."""
        value = self._get_raw_tensor("replicate_counts")
        if value is None:
            raise RuntimeError("replicate_counts was unexpectedly stored as None.")
        return value

    @property
    def replicate_positive_counts(self) -> Tensor:
        """Return positive-label counts for each replicated condition."""
        value = self._get_raw_tensor("replicate_positive_counts")
        if value is None:
            raise RuntimeError("replicate_positive_counts was unexpectedly stored as None.")
        return value

    @property
    def replicate_positive_rate(self) -> Tensor:
        """Return empirical positive-label rates without using them as training targets."""
        return self.replicate_positive_counts / self.replicate_counts.to(
            dtype=self.replicate_positive_counts.dtype,
        )

    @property
    def replicate_disagreement(self) -> Tensor:
        """Return pairwise label-disagreement probability for each condition."""
        value = self._get_raw_tensor("replicate_disagreement")
        if value is None:
            raise RuntimeError("replicate_disagreement was unexpectedly stored as None.")
        return value

    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        """Return repeated-label robustness metadata."""
        return frozenset({ClassificationRobustnessType.REPLICATE_LABELS})
