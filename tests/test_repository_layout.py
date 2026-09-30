"""Repository layout contracts that do not require importing the package."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "robotorchan"


def test_model_tests_are_owned_by_a_family_or_contract_directory() -> None:
    model_test_root = REPOSITORY_ROOT / "tests" / "models"
    misplaced = sorted(path.name for path in model_test_root.glob("test_*.py"))

    assert not misplaced, (
        f"Model tests must live in a model family, contracts, or integration directory: {misplaced}"
    )


def test_optimizer_benchmarks_are_owned_by_benchmark_package() -> None:
    assert not (SOURCE_ROOT / "optim" / "benchmark.py").exists()
    assert (SOURCE_ROOT / "benchmarks" / "optimization.py").is_file()


def test_capability_workflow_is_owned_by_one_package() -> None:
    for old_module in ("problem.py", "selector.py", "recommendation.py"):
        assert not (SOURCE_ROOT / old_module).exists()

    workflow_root = SOURCE_ROOT / "workflow"
    assert {path.name for path in workflow_root.glob("*.py")} == {
        "__init__.py",
        "problem.py",
        "recommendation.py",
        "selector.py",
    }

    test_root = REPOSITORY_ROOT / "tests"
    workflow_test_names = {
        "test_capability_workflow.py",
        "test_problem.py",
        "test_problem_contracts.py",
        "test_problem_objective_output_contract.py",
        "test_recommendation.py",
        "test_selector.py",
    }
    assert not any((test_root / name).exists() for name in workflow_test_names)
    actual_workflow_tests = {path.name for path in (test_root / "workflow").glob("test_*.py")}
    assert workflow_test_names <= actual_workflow_tests


def test_concrete_optimizer_strategies_are_owned_by_strategy_package() -> None:
    strategy_modules = {"mixed.py", "original.py", "random.py", "sobol.py", "tree.py"}
    optim_root = SOURCE_ROOT / "optim"
    assert not any((optim_root / name).exists() for name in strategy_modules)

    strategy_root = optim_root / "strategies"
    assert strategy_modules | {"__init__.py"} == {path.name for path in strategy_root.glob("*.py")}

    optim_test_root = REPOSITORY_ROOT / "tests" / "optim"
    strategy_tests = {
        f"test_{name.removesuffix('.py')}_space_strategy.py" for name in ("mixed.py", "original.py")
    }
    strategy_tests |= {
        "test_random_search_strategy.py",
        "test_sobol_search_strategy.py",
        "test_tree_ensemble_search_strategy.py",
    }
    assert not any((optim_test_root / name).exists() for name in strategy_tests)
    actual_strategy_tests = set(
        path.name for path in (optim_test_root / "strategies").glob("test_*.py")
    )
    assert strategy_tests <= actual_strategy_tests


def test_candidate_constraints_are_owned_by_constraint_package() -> None:
    optim_root = SOURCE_ROOT / "optim"
    assert not (optim_root / "constraint_evaluation.py").exists()
    assert (optim_root / "constraints").is_dir()

    constraint_modules = {path.name for path in (optim_root / "constraints").glob("*.py")}
    assert constraint_modules == {"__init__.py", "contracts.py", "evaluation.py"}


def test_cross_backend_optimizer_contracts_have_dedicated_ownership() -> None:
    contract_test_names = {
        "test_optimizer_capability_validation.py",
        "test_optimizer_correctness_regressions.py",
        "test_optimizer_dispatch.py",
        "test_search_strategy_base.py",
    }
    optim_test_root = REPOSITORY_ROOT / "tests" / "optim"
    assert not any((optim_test_root / name).exists() for name in contract_test_names)

    actual_contract_tests = set(
        path.name for path in (optim_test_root / "contracts").glob("test_*.py")
    )
    assert contract_test_names <= actual_contract_tests


def test_optimizer_e2e_tests_have_integration_ownership() -> None:
    integration_test_names = {
        "test_optimizer_model_acquisition_e2e.py",
        "test_sampling_mc_acquisition_e2e.py",
        "test_sampling_multiobjective_constrained_e2e.py",
    }
    optim_test_root = REPOSITORY_ROOT / "tests" / "optim"
    assert not any((optim_test_root / name).exists() for name in integration_test_names)

    actual_integration_tests = set(
        path.name for path in (optim_test_root / "integration").glob("test_*.py")
    )
    assert integration_test_names <= actual_integration_tests


def test_one_shot_optimization_has_dedicated_ownership() -> None:
    optim_root = SOURCE_ROOT / "optim"
    assert not (optim_root / "initializers.py").exists()
    assert not (optim_root / "mixed_one_shot.py").exists()

    one_shot_root = optim_root / "one_shot"
    assert {path.name for path in one_shot_root.glob("*.py")} == {
        "__init__.py",
        "initialization.py",
        "mixed.py",
    }

    optim_test_root = REPOSITORY_ROOT / "tests" / "optim"
    assert not (optim_test_root / "test_mixed_one_shot.py").exists()
    assert (optim_test_root / "one_shot" / "test_mixed.py").is_file()
