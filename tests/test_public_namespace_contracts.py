"""Public namespace contracts for the reorganized package."""

from __future__ import annotations

import importlib

import robotorchan.models as models
import robotorchan.optim as optim
import robotorchan.reduction as reduction


def test_public_namespaces_import() -> None:
    for module_name in (
        "robotorchan.acquisition",
        "robotorchan.models",
        "robotorchan.models.expressive",
        "robotorchan.models.high_dimensional",
        "robotorchan.models.high_dimensional.reduced",
        "robotorchan.models.preference",
        "robotorchan.models.robust",
        "robotorchan.models.standard",
        "robotorchan.models.structured",
        "robotorchan.models.uncertain",
        "robotorchan.objectives",
        "robotorchan.optim",
        "robotorchan.reduction",
    ):
        assert importlib.import_module(module_name).__name__ == module_name


def test_reduced_models_are_canonical_model_exports() -> None:
    reduced = importlib.import_module("robotorchan.models.high_dimensional.reduced")

    for name in reduced.__all__:
        assert getattr(models, name) is getattr(reduced, name)


def test_reduction_exports_resolve() -> None:
    for name in reduction.__all__:
        assert getattr(reduction, name) is not None


def test_optim_exports_resolve() -> None:
    for name in optim.__all__:
        assert getattr(optim, name) is not None


def test_removed_module_paths_do_not_import() -> None:
    removed_modules = (
        "robotorchan.models.reduction",
        "robotorchan.models.neural_reduction",
        "robotorchan.models.output_reduction",
        "robotorchan.models.supervised_neural_reduction",
        "robotorchan.models.supervised_neural",
        "robotorchan.models.joint_neural",
        "robotorchan.models.joint_vae",
        "robotorchan.models.vae",
        "robotorchan.models.pairwise",
        "robotorchan.models.robust_models",
        "robotorchan.models.uncertain_categorical",
        "robotorchan.models.uncertain_input",
    )
    for module_name in removed_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"removed module path remains importable: {module_name}")

def test_model_family_exports_are_canonical() -> None:
    family_modules = (
        "robotorchan.models.expressive",
        "robotorchan.models.high_dimensional",
        "robotorchan.models.preference",
        "robotorchan.models.robust",
        "robotorchan.models.standard",
        "robotorchan.models.structured",
        "robotorchan.models.uncertain",
    )
    for module_name in family_modules:
        family = importlib.import_module(module_name)
        for name in family.__all__:
            assert getattr(models, name) is getattr(family, name)
