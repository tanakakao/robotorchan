"""Import-contract tests for the binary classification package layout."""

from robotorchan.models.classification import BinarySingleTaskGPClassifier as PublicBinary
from robotorchan.models.classification.binary import BinarySingleTaskGPClassifier
from robotorchan.models.classification.binary.standard.multitask import (
    KroneckerMultiTaskBinaryGPClassifier,
    MultiTaskBinaryGPClassifier,
)
from robotorchan.models.classification.binary.standard.single_task import (
    MixedBinarySingleTaskGPClassifier,
)


def test_binary_namespace_preserves_public_class_identity() -> None:
    """The package and public namespace must expose the same model class."""
    assert BinarySingleTaskGPClassifier is PublicBinary


def test_binary_standard_layout_exposes_single_and_multitask_models() -> None:
    """Regression-aligned binary modules must expose the expected families."""
    assert MixedBinarySingleTaskGPClassifier.__name__ == "MixedBinarySingleTaskGPClassifier"
    assert MultiTaskBinaryGPClassifier.__name__ == "MultiTaskBinaryGPClassifier"
    assert (
        KroneckerMultiTaskBinaryGPClassifier.__name__
        == "KroneckerMultiTaskBinaryGPClassifier"
    )
