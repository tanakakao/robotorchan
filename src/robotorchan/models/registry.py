"""Initial model capability registry."""

from robotorchan.models.capabilities import (
    DocumentationLinks,
    HighDimensionalStrategy,
    InferenceType,
    InputType,
    ModelCapabilities,
    ModelRegistryEntry,
    RobustnessType,
    TaskType,
)


def _docs(guide: str, theory: str, notebook: str) -> DocumentationLinks:
    return DocumentationLinks(guide, theory, notebook)


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "SingleTaskGP": ModelRegistryEntry(
        "SingleTaskGP", ModelCapabilities(),
        _docs("docs/models/standard.md", "docs/theory/02_gaussian_process.md",
              "examples/notebooks/01_single_task_gp.ipynb"),
        "BoTorch exact single-task GP wrapper"),
    "MixedSingleTaskGP": ModelRegistryEntry(
        "MixedSingleTaskGP", ModelCapabilities(input_type=InputType.MIXED),
        _docs("docs/models/standard.md", "docs/theory/05_mixed_variables.md",
              "examples/notebooks/02_mixed_single_task_gp.ipynb"),
        "native mixed-variable GP path"),
    "KroneckerMultiTaskGP": ModelRegistryEntry(
        "KroneckerMultiTaskGP",
        ModelCapabilities(task_type=TaskType.MULTITASK),
        _docs("docs/models/multitask_multioutput.md", "docs/theory/07_multitask_multioutput.md",
              "examples/notebooks/04_multitask_gp.ipynb"),
        "Kronecker-structured exact multitask GP"),
    "MixedSaasFullyBayesianMultiTaskGP": ModelRegistryEntry(
        "MixedSaasFullyBayesianMultiTaskGP",
        ModelCapabilities(input_type=InputType.MIXED, task_type=TaskType.MULTITASK,
                          inference=InferenceType.FULLY_BAYESIAN,
                          high_dimensional=HighDimensionalStrategy.SAAS),
        _docs("docs/models/multitask_multioutput.md",
              "docs/theory/21_advanced_high_dimensional_models.md",
              "examples/notebooks/08_saas_gp.ipynb"),
        "fully Bayesian SAAS multitask model with mixed inputs"),
    "PCAGP": ModelRegistryEntry(
        "PCAGP", ModelCapabilities(high_dimensional=HighDimensionalStrategy.REDUCTION),
        _docs("docs/models/reduced.md", "docs/theory/18_dimensionality_reduction_gp.md",
              "examples/notebooks/19_reduced_gp.ipynb"),
        "PCA reduction followed by an exact GP"),
    "MixedRobustRelevancePursuitMultiTaskGP": ModelRegistryEntry(
        "MixedRobustRelevancePursuitMultiTaskGP",
        ModelCapabilities(input_type=InputType.MIXED, task_type=TaskType.MULTITASK,
                          robustness=frozenset({RobustnessType.RELEVANCE_PURSUIT})),
        _docs("docs/models/robust_noise.md", "docs/theory/14_robust_gaussian_process.md",
              "examples/notebooks/10_robust_gp.ipynb"),
        "robust relevance-pursuit multitask GP with mixed inputs"),
    "SingleTaskMultiFidelityGP": ModelRegistryEntry(
        "SingleTaskMultiFidelityGP", ModelCapabilities(multi_fidelity=True),
        _docs("docs/models/standard.md", "docs/theory/06_multi_fidelity.md",
              "examples/notebooks/03_multi_fidelity_gp.ipynb"),
        "BoTorch single-task multi-fidelity GP wrapper"),
    "SingleTaskVariationalGP": ModelRegistryEntry(
        "SingleTaskVariationalGP", ModelCapabilities(inference=InferenceType.VARIATIONAL),
        _docs("docs/models/standard.md", "docs/theory/09_variational_gp.md",
              "examples/notebooks/06_variational_gp.ipynb"),
        "variational single-task GP"),
}


def get_model_registry_entry(model_name: str) -> ModelRegistryEntry:
    """Return registry metadata for a registered public model."""
    return MODEL_REGISTRY[model_name]
