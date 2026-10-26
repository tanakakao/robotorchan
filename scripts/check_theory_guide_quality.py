"""Check the fixed 48-chapter Japanese theory-to-implementation guide layout."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = Path("docs/development/theory-implementation")
THEORY = Path("docs/theory")
MAIN_CHAPTERS = (
    "01_bayesian_optimization", "02_gaussian_process", "03_kernel",
    "04_acquisition_function", "05_mixed_variables", "06_multi_fidelity",
    "07_multitask_multioutput", "08_high_dimensional_gp", "09_variational_gp",
    "10_preference_learning", "11_structured_output", "12_hierarchical_contextual_gp",
    "13_model_selection", "14_robust_gaussian_process", "15_heteroskedastic_noise",
    "16_uncertain_input_gp", "17_nonstationary_gp", "18_dimensionality_reduction_gp",
    "19_neural_reduction_gp", "20_high_dimensional_search",
    "21_advanced_high_dimensional_models", "22_expressive_gp",
    "23_non_gp_surrogates", "24_classification",
)
ACQUISITION_CHAPTERS = (
    "01_foundations", "02_improvement", "03_confidence_bound",
    "04_batch_noisy", "05_information_theoretic", "06_lookahead",
    "07_multiobjective", "08_constraints", "09_active_learning",
    "10_level_set", "11_multifidelity_cost_aware", "12_selection_guide",
)
OPTIMIZATION_CHAPTERS = (
    "01_foundations", "02_gradient_based", "03_initialization",
    "04_mixed_discrete", "05_constraints", "06_derivative_free",
    "07_evolutionary", "08_batch_one_shot", "09_high_dimensional",
    "10_trust_region", "11_hybrid", "12_selection_guide",
)
MAPPING_ORIGINALS = (
    "README.md", "models.md", "acquisition.md", "optimization.md", "verification.md",
)


def expected_chapters() -> tuple[Path, ...]:
    """Return theory-relative chapter paths, preserving the fixed plan."""
    return tuple(
        [Path(f"{name}.md") for name in MAIN_CHAPTERS]
        + [Path("acquisition") / f"{name}.md" for name in ACQUISITION_CHAPTERS]
        + [Path("optimization") / f"{name}.md" for name in OPTIMIZATION_CHAPTERS]
    )


def check_guide_quality(root: Path = ROOT) -> list[str]:
    """Check structural correspondence without making mathematical claims."""
    failures: list[str] = []
    guide = root / MAPPING / "guide"
    theory = root / THEORY
    planned = expected_chapters()
    if len(planned) != 48 or len(set(planned)) != 48:
        failures.append("fixed 48-chapter contract is inconsistent")
    for name in MAPPING_ORIGINALS:
        if not (root / MAPPING / name).is_file():
            failures.append(f"missing original mapping: {name}")
    for relative in planned:
        source = theory / relative
        target = guide / relative
        if not source.is_file():
            failures.append(f"missing canonical theory: {relative.as_posix()}")
        if not target.is_file():
            failures.append(f"missing guide chapter: {relative.as_posix()}")
            continue
        content = target.read_text(encoding="utf-8")
        if not re.search(r"(?m)^#\s+\S", content):
            failures.append(f"missing chapter heading: {relative.as_posix()}")
        if not re.search(r"(?m)^##\s+\S", content):
            failures.append(f"missing explanatory sections: {relative.as_posix()}")
        if not re.search(r"\]\([^)]*theory/", content):
            failures.append(f"missing theory navigation: {relative.as_posix()}")
    return failures


def main() -> int:
    """Report structural failures for CI."""
    failures = check_guide_quality()
    for failure in failures:
        print(failure)
    if failures:
        return 1
    print("All 48 theory-aligned guide chapters and original Mapping files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
