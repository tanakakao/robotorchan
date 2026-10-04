"""Multiclass-readiness contracts for the classification common layer."""

import pytest
import torch

from robotorchan.models.classification import (
    ClassificationLikelihoodFamily,
    ClassificationMetadata,
    ClassificationModelMixin,
    LatentOutputStructure,
    list_classification_models,
)
from robotorchan.models.classification.validation import validate_class_indices


def test_common_metadata_represents_multiclass_classification() -> None:
    metadata = ClassificationMetadata(
        num_classes=3,
        class_labels=("alpha", "beta", "gamma"),
        likelihood_family=ClassificationLikelihoodFamily.CATEGORICAL,
        latent_output_structure=LatentOutputStructure.PER_CLASS,
    )

    assert metadata.num_classes == 3
    assert metadata.class_labels == ("alpha", "beta", "gamma")
    assert metadata.likelihood_family is ClassificationLikelihoodFamily.CATEGORICAL
    assert metadata.latent_output_structure is LatentOutputStructure.PER_CLASS


def test_common_model_contract_does_not_define_binary_threshold() -> None:
    assert "threshold" not in ClassificationModelMixin.predict_class.__annotations__


def test_validate_class_indices_accepts_canonical_multiclass_labels() -> None:
    labels = torch.tensor([0, 2, 1, 2], dtype=torch.long)

    validated = validate_class_indices(labels, num_classes=3)

    assert validated is labels


@pytest.mark.parametrize(
    ("labels", "num_classes", "error_type", "match"),
    [
        (torch.tensor([0, 3]), 3, ValueError, r"\[0, 3\)"),
        (torch.tensor([-1, 1]), 3, ValueError, r"\[0, 3\)"),
        (torch.tensor([0.0, 1.0]), 3, TypeError, "integer dtype"),
        (torch.tensor([0, 1]), 1, ValueError, "at least 2"),
    ],
)
def test_validate_class_indices_rejects_noncanonical_labels(
    labels: torch.Tensor,
    num_classes: int,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        validate_class_indices(labels, num_classes=num_classes)


def test_registry_namespace_is_ready_for_future_multiclass_ids() -> None:
    assert list_classification_models(prefix="binary.")
    assert list_classification_models(prefix="multiclass.") == ()
