"""Tests for the uncertain binary-classification contract."""

from robotorchan.models.classification.binary.uncertain import (
    ClassificationInputUncertaintyType,
    ClassificationUncertaintyIntegration,
    ClassificationUncertaintyTarget,
    UncertainBinaryClassificationMixin,
)


class _ExampleUncertainClassifier(UncertainBinaryClassificationMixin):
    @property
    def classification_input_uncertainty(
        self,
    ) -> frozenset[ClassificationInputUncertaintyType]:
        return frozenset({ClassificationInputUncertaintyType.CONTINUOUS})

    @property
    def classification_uncertainty_target(self) -> ClassificationUncertaintyTarget:
        return ClassificationUncertaintyTarget.CANDIDATE_INPUTS

    @property
    def classification_uncertainty_integration(
        self,
    ) -> ClassificationUncertaintyIntegration:
        return ClassificationUncertaintyIntegration.MONTE_CARLO


def test_uncertain_classification_metadata_is_explicit() -> None:
    model = _ExampleUncertainClassifier()

    metadata = model.uncertain_classification_metadata

    assert model.is_uncertain_classification is True
    assert metadata.uncertainty == frozenset({ClassificationInputUncertaintyType.CONTINUOUS})
    assert metadata.target is ClassificationUncertaintyTarget.CANDIDATE_INPUTS
    assert metadata.integration is ClassificationUncertaintyIntegration.MONTE_CARLO
    assert metadata.preserves_latent_classification_posterior is True


def test_uncertain_classification_enums_are_multiclass_neutral() -> None:
    assert set(ClassificationInputUncertaintyType) == {
        ClassificationInputUncertaintyType.CONTINUOUS,
        ClassificationInputUncertaintyType.CATEGORICAL,
    }
    assert set(ClassificationUncertaintyTarget) == {
        ClassificationUncertaintyTarget.TRAINING_INPUTS,
        ClassificationUncertaintyTarget.CANDIDATE_INPUTS,
        ClassificationUncertaintyTarget.BOTH,
    }
    assert set(ClassificationUncertaintyIntegration) == {
        ClassificationUncertaintyIntegration.ANALYTIC,
        ClassificationUncertaintyIntegration.MONTE_CARLO,
        ClassificationUncertaintyIntegration.QUASI_MONTE_CARLO,
    }
