"""Tests for the reviewed regression-to-classification family mapping."""

import pytest

from robotorchan.models.classification.regression_family_audit import (
    REGRESSION_FAMILY_CLASSIFICATION_AUDIT,
    ClassificationPortStatus,
    classification_port_status,
)


def test_audit_has_unique_family_names() -> None:
    families = [item.family for item in REGRESSION_FAMILY_CLASSIFICATION_AUDIT]
    assert len(families) == len(set(families))


@pytest.mark.parametrize(
    ("family", "status"),
    [
        ("spectral_mixture", ClassificationPortStatus.NATURAL_FOLLOW_UP),
        ("infinite_width_bnn", ClassificationPortStatus.NATURAL_FOLLOW_UP),
        ("nonstationary", ClassificationPortStatus.NATURAL_FOLLOW_UP),
        (
            "student_t_and_contaminated_noise",
            ClassificationPortStatus.NOT_DIRECTLY_APPLICABLE,
        ),
        (
            "heteroskedastic_and_replicate_noise",
            ClassificationPortStatus.NOT_DIRECTLY_APPLICABLE,
        ),
    ],
)
def test_reviewed_family_status(family: str, status: ClassificationPortStatus) -> None:
    assert classification_port_status(family) is status


def test_unknown_family_is_not_silently_classified() -> None:
    with pytest.raises(KeyError, match="Unknown audited regression family"):
        classification_port_status("unknown")
