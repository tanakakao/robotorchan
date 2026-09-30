"""Public namespace contracts for the reorganized package."""

from __future__ import annotations

import importlib

import robotorchan.acquisition as acquisition
import robotorchan.benchmarks as benchmarks
import robotorchan.models as models
import robotorchan.optim as optim
import robotorchan.optim.backend_support as backend_support
import robotorchan.optim.constraints as constraints
import robotorchan.optim.domains as domains
import robotorchan.optim.one_shot as one_shot
import robotorchan.optim.strategies as strategies
import robotorchan.reduction as reduction
import robotorchan.uncertainty as uncertainty
import robotorchan.workflow as workflow


def test_public_namespaces_import() -> None:
    for module_name in (
        "robotorchan.acquisition",
        "robotorchan.benchmarks",
        "robotorchan.models",
        "robotorchan.models.expressive",
        "robotorchan.models.high_dimensional",
        "robotorchan.models.non_gp",
        "robotorchan.models.high_dimensional.reduced",
        "robotorchan.models.preference",
        "robotorchan.models.robust",
        "robotorchan.models.standard",
        "robotorchan.models.structured",
        "robotorchan.models.uncertain",
        "robotorchan.objectives",
        "robotorchan.optim",
        "robotorchan.optim.backend_support",
        "robotorchan.optim.constraints",
        "robotorchan.optim.domains",
        "robotorchan.optim.one_shot",
        "robotorchan.optim.strategies",
        "robotorchan.reduction",
        "robotorchan.uncertainty",
        "robotorchan.workflow",
    ):
        assert importlib.import_module(module_name).__name__ == module_name


def test_reduced_models_are_canonical_model_exports() -> None:
    reduced = importlib.import_module("robotorchan.models.high_dimensional.reduced")

    for name in reduced.__all__:
        assert getattr(models, name) is getattr(reduced, name)


def test_acquisition_exports_resolve() -> None:
    for name in acquisition.__all__:
        assert getattr(acquisition, name) is not None


def test_benchmark_exports_resolve() -> None:
    for name in benchmarks.__all__:
        assert getattr(benchmarks, name) is not None


def test_workflow_exports_resolve() -> None:
    for name in workflow.__all__:
        assert getattr(workflow, name) is not None


def test_uncertainty_exports_resolve() -> None:
    for name in uncertainty.__all__:
        assert getattr(uncertainty, name) is not None


def test_reduction_exports_resolve() -> None:
    for name in reduction.__all__:
        assert getattr(reduction, name) is not None


def test_backend_support_exports_are_canonical() -> None:
    for name in backend_support.__all__:
        assert getattr(optim, name) is getattr(backend_support, name)


def test_constraint_exports_are_canonical() -> None:
    for name in constraints.__all__:
        assert getattr(optim, name) is getattr(constraints, name)


def test_domain_exports_are_canonical() -> None:
    for name in domains.__all__:
        assert getattr(optim, name) is getattr(domains, name)


def test_one_shot_exports_are_canonical() -> None:
    for name in one_shot.__all__:
        assert getattr(optim, name) is getattr(one_shot, name)


def test_strategy_exports_are_canonical() -> None:
    for name in strategies.__all__:
        assert getattr(optim, name) is getattr(strategies, name)


def test_optim_exports_resolve() -> None:
    for name in optim.__all__:
        assert getattr(optim, name) is not None


def test_removed_module_paths_do_not_import() -> None:
    removed_modules = (
        "robotorchan.acquisition.mixed_one_shot",
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
        "robotorchan.optim.constraint_evaluation",
        "robotorchan.optim.cross_cutting",
        "robotorchan.optim.initializers",
        "robotorchan.optim.variable_space",
        "robotorchan.optim.mixed",
        "robotorchan.optim.mixed_one_shot",
        "robotorchan.optim.original",
        "robotorchan.optim.runtime",
        "robotorchan.optim.random",
        "robotorchan.optim.sobol",
        "robotorchan.optim.tree",
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
        "robotorchan.models.non_gp",
        "robotorchan.models.preference",
        "robotorchan.models.robust",
        "robotorchan.models.standard",
        "robotorchan.models.structured",
        "robotorchan.models.uncertain",
    )
    for module_name in family_modules:
        family = importlib.import_module(module_name)
        canonical_names = set(family.__all__) & set(models.__all__)
        for name in canonical_names:
            assert getattr(models, name) is getattr(family, name)


def test_all_top_level_models_are_owned_by_a_family_package() -> None:
    family_modules = (
        "robotorchan.models.expressive",
        "robotorchan.models.high_dimensional",
        "robotorchan.models.non_gp",
        "robotorchan.models.preference",
        "robotorchan.models.robust",
        "robotorchan.models.standard",
        "robotorchan.models.structured",
        "robotorchan.models.uncertain",
    )
    family_names: set[str] = set()
    for module_name in family_modules:
        family = importlib.import_module(module_name)
        family_names.update(
            name for name in family.__all__ if isinstance(getattr(family, name), type)
        )

    top_level_model_names = {
        name for name in models.__all__ if isinstance(getattr(models, name), type)
    }
    top_level_model_names.discard("UnsupportedModelOperationError")

    assert top_level_model_names <= family_names
