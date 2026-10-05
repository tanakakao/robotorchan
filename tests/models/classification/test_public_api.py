"""Public import contracts for classification surrogate models."""

import importlib

import pytest

import robotorchan.models.classification as classification
import robotorchan.models.classification.binary as binary

PUBLIC_BINARY_MODEL_NAMES = (
    "ALEBOBinarySingleTaskGPClassifier",
    "BinarySingleTaskDeepGPClassifier",
    "BinarySingleTaskGPClassifier",
    "JointEncoderBinaryGPClassifier",
    "KroneckerMultiTaskBinaryGPClassifier",
    "MapSaasBinarySingleTaskGPClassifier",
    "MixedBinarySingleTaskGPClassifier",
    "MultiTaskBinaryGPClassifier",
    "PCABinarySingleTaskGPClassifier",
    "PLSBinarySingleTaskGPClassifier",
    "RandomProjectionBinarySingleTaskGPClassifier",
    "ReducedBinarySingleTaskGPClassifier",
    "SaasBinarySingleTaskGPClassifier",
)


def test_classification_and_binary_facades_share_public_binary_classes() -> None:
    for name in PUBLIC_BINARY_MODEL_NAMES:
        assert name in classification.__all__
        assert name in binary.__all__
        assert getattr(classification, name) is getattr(binary, name)


def test_family_facades_resolve_moved_models_to_same_public_classes() -> None:
    expressive = importlib.import_module(
        "robotorchan.models.classification.binary.expressive"
    )
    reduced = importlib.import_module(
        "robotorchan.models.classification.binary.high_dimensional.reduced"
    )

    assert (
        expressive.BinarySingleTaskDeepGPClassifier
        is binary.BinarySingleTaskDeepGPClassifier
    )
    assert (
        reduced.JointEncoderBinaryGPClassifier
        is binary.JointEncoderBinaryGPClassifier
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "robotorchan.models.classification.binary.high_dimensional.deep_gp",
        "robotorchan.models.classification.binary.high_dimensional.joint_neural",
    ),
)
def test_removed_internal_module_paths_are_not_compatibility_shims(
    module_name: str,
) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
