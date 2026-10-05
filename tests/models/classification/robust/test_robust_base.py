"""Tests for robust binary-classification contracts."""

from robotorchan.models.classification.binary.robust import (
    ClassificationRobustnessType,
    RobustBinaryClassificationMixin,
    RobustClassificationMetadata,
)


class _RobustStub(RobustBinaryClassificationMixin):
    @property
    def classification_robustness(self) -> frozenset[ClassificationRobustnessType]:
        return frozenset({ClassificationRobustnessType.LABEL_NOISE})

    @property
    def models_observed_label_process(self) -> bool:
        return True


def test_robust_metadata_is_classification_native() -> None:
    model = _RobustStub()

    assert model.is_robust_classification
    assert model.classification_robustness == frozenset(
        {ClassificationRobustnessType.LABEL_NOISE}
    )
    assert model.robust_classification_metadata == RobustClassificationMetadata(
        robustness=frozenset({ClassificationRobustnessType.LABEL_NOISE}),
        models_observed_label_process=True,
        preserves_latent_classification_posterior=True,
    )


def test_robustness_types_cover_planned_binary_robust_families() -> None:
    assert set(ClassificationRobustnessType) == {
        ClassificationRobustnessType.LABEL_NOISE,
        ClassificationRobustnessType.CONTAMINATION,
        ClassificationRobustnessType.REPLICATE_LABELS,
        ClassificationRobustnessType.INPUT_DEPENDENT_LABEL_NOISE,
        ClassificationRobustnessType.NONSTATIONARY,
    }
