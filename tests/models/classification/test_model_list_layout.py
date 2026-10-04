"""Import-contract tests for shared classification composition."""

from robotorchan.models.classification import ClassificationModelList as PublicModelList
from robotorchan.models.classification.model_list import ClassificationModelList


def test_classification_model_list_has_shared_canonical_owner() -> None:
    """The public class must be owned by the classification-wide module."""
    assert ClassificationModelList is PublicModelList


def test_classification_model_list_is_not_binary_specific() -> None:
    """The composition class must not be exported from binary namespaces."""
    import robotorchan.models.classification.binary as binary
    import robotorchan.models.classification.binary.standard as standard

    assert not hasattr(binary, "ClassificationModelList")
    assert not hasattr(standard, "ClassificationModelList")
