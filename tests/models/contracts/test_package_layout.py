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


def test_classification_common_layer_contains_only_shared_modules() -> None:
    common_modules = {
        path.name
        for path in CLASSIFICATION_ROOT.glob("*.py")
        if path.name != "__init__.py"
    }

    assert common_modules == {
        "base.py",
        "model_list.py",
        "registry.py",
        "validation.py",
    }


def test_development_audits_are_not_runtime_modules() -> None:
    assert not (CLASSIFICATION_ROOT / "regression_family_audit.py").exists()
