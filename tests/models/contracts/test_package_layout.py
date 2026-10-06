"""Final filesystem contracts for the surrogate-model package."""

from pathlib import Path

import robotorchan.models as models_package

MODELS_ROOT = Path(models_package.__file__).parent
CLASSIFICATION_ROOT = MODELS_ROOT / "classification"
BINARY_ROOT = CLASSIFICATION_ROOT / "binary"

REGRESSION_FAMILY_DIRECTORIES = frozenset(
    {
        "expressive",
        "high_dimensional",
        "non_gp",
        "preference",
        "robust",
        "standard",
        "structured",
        "uncertain",
    }
)


def test_regression_family_directories_remain_top_level() -> None:
    existing = {
        path.name
        for path in MODELS_ROOT.iterdir()
        if path.is_dir() and path.name in REGRESSION_FAMILY_DIRECTORIES
    }

    assert existing == REGRESSION_FAMILY_DIRECTORIES
    assert not (MODELS_ROOT / "regression").exists()


def test_classification_task_axis_is_explicit() -> None:
    assert BINARY_ROOT.is_dir()
    assert not (CLASSIFICATION_ROOT / "standard").exists()
    assert not (CLASSIFICATION_ROOT / "high_dimensional").exists()


def test_binary_model_families_are_nested_under_binary_task() -> None:
    assert (BINARY_ROOT / "standard").is_dir()
    assert (BINARY_ROOT / "high_dimensional").is_dir()
    assert (BINARY_ROOT / "expressive").is_dir()


def test_classification_common_layer_contains_only_shared_modules() -> None:
    common_modules = {
        path.name for path in CLASSIFICATION_ROOT.glob("*.py") if path.name != "__init__.py"
    }

    assert common_modules == {
        "base.py",
        "model_list.py",
        "probability.py",
        "registry.py",
        "validation.py",
    }


def test_development_audits_are_not_runtime_modules() -> None:
    assert not (CLASSIFICATION_ROOT / "regression_family_audit.py").exists()
    assert not (CLASSIFICATION_ROOT / "structured_audit.py").exists()


def test_binary_deep_gp_is_owned_by_expressive_family() -> None:
    assert (BINARY_ROOT / "expressive" / "deep_gp.py").is_file()
    assert not (BINARY_ROOT / "high_dimensional" / "deep_gp.py").exists()


def test_binary_joint_encoder_is_owned_by_reduced_family() -> None:
    reduced_root = BINARY_ROOT / "high_dimensional" / "reduced"
    assert (reduced_root / "joint_neural.py").is_file()
    assert not (BINARY_ROOT / "high_dimensional" / "joint_neural.py").exists()


def test_classification_family_tests_follow_model_ownership() -> None:
    test_root = MODELS_ROOT.parent.parent.parent / "tests" / "models" / "classification"

    assert (test_root / "expressive" / "test_deep_gp_classifier.py").is_file()
    assert (
        test_root / "high_dimensional" / "reduced" / "test_joint_encoder_classifier.py"
    ).is_file()
    assert not (test_root / "high_dimensional" / "test_deep_gp_classifier.py").exists()
    assert not (test_root / "high_dimensional" / "test_joint_encoder_classifier.py").exists()
