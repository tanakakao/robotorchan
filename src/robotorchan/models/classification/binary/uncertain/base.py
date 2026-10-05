"""Common contracts for binary classification with uncertain inputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class ClassificationInputUncertaintyType(StrEnum):
    """Kinds of input uncertainty represented by a classifier."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class ClassificationUncertaintyTarget(StrEnum):
    """Stage at which input uncertainty is integrated."""

    TRAINING_INPUTS = "training_inputs"
    CANDIDATE_INPUTS = "candidate_inputs"
    BOTH = "both"


class ClassificationUncertaintyIntegration(StrEnum):
    """Numerical or analytic strategy used to integrate input uncertainty."""

    ANALYTIC = "analytic"
    MONTE_CARLO = "monte_carlo"
    QUASI_MONTE_CARLO = "quasi_monte_carlo"


@dataclass(frozen=True, slots=True)
class UncertainClassificationMetadata:
    """Stable metadata describing classification input-uncertainty semantics."""

    uncertainty: frozenset[ClassificationInputUncertaintyType]
    target: ClassificationUncertaintyTarget
    integration: ClassificationUncertaintyIntegration
    preserves_latent_classification_posterior: bool


class UncertainBinaryClassificationMixin:
    """Semantic contract shared by binary classifiers with uncertain inputs."""

    is_uncertain_classification: ClassVar[bool] = True

    @property
    def classification_input_uncertainty(
        self,
    ) -> frozenset[ClassificationInputUncertaintyType]:
        """Return the represented input-uncertainty mechanisms."""
        raise NotImplementedError

    @property
    def classification_uncertainty_target(self) -> ClassificationUncertaintyTarget:
        """Return where input uncertainty is integrated."""
        raise NotImplementedError

    @property
    def classification_uncertainty_integration(
        self,
    ) -> ClassificationUncertaintyIntegration:
        """Return the integration strategy."""
        raise NotImplementedError

    @property
    def preserves_latent_classification_posterior(self) -> bool:
        """Return whether the base latent class-function posterior is unchanged."""
        return True

    @property
    def uncertain_classification_metadata(self) -> UncertainClassificationMetadata:
        """Return immutable uncertain-classification metadata."""
        return UncertainClassificationMetadata(
            uncertainty=self.classification_input_uncertainty,
            target=self.classification_uncertainty_target,
            integration=self.classification_uncertainty_integration,
            preserves_latent_classification_posterior=(
                self.preserves_latent_classification_posterior
            ),
        )
