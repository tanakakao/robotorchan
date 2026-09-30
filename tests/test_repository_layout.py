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
