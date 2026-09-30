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
    assert workflow_test_names <= {
        path.name for path in (test_root / "workflow").glob("test_*.py")
    }
