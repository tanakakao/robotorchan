"""Common contracts for robust binary classification models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class ClassificationRobustnessType(StrEnum):
    """Classification-native robustness mechanisms."""

    LABEL_NOISE = "label_noise"
    CONTAMINATION = "contamination"
    REPLICATE_LABELS = "replicate_labels"
    INPUT_DEPENDENT_LABEL_NOISE = "input_dependent_label_noise"
    NONSTATIONARY = "nonstationary"


@dataclass(frozen=True, slots=True)
class RobustClassificationMetadata:
    """Stable metadata describing robust-classification semantics."""

    robustness: frozenset[ClassificationRobustnessType]
    models_observed_label_process: bool
    preserves_latent_classification_posterior: bool


class RobustBinaryClassificationMixin:
    """Semantic contract shared by robust binary classifiers."""

    is_robust_classification: ClassVar[bool] = True

    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        """Return the classification-native robustness mechanisms."""
        raise NotImplementedError

    @property
    def models_observed_label_process(self) -> bool:
        """Return whether observed-label corruption is explicitly represented."""
        return False

    @property
    def preserves_latent_classification_posterior(self) -> bool:
        """Return whether posterior remains the clean latent class-function posterior."""
        return True

    @property
    def robust_classification_metadata(self) -> RobustClassificationMetadata:
        """Return immutable robust-classification metadata."""
        return RobustClassificationMetadata(
            robustness=self.classification_robustness,
            models_observed_label_process=self.models_observed_label_process,
            preserves_latent_classification_posterior=(
                self.preserves_latent_classification_posterior
            ),
        )
