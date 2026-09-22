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
        "SingleTaskGP",
        ModelCapabilities(),
        _docs(
            "docs/models/standard.md",
            "docs/theory/02_gaussian_process.md",
            "examples/notebooks/01_single_task_gp.ipynb",
        ),
        "BoTorch exact single-task GP wrapper",
    ),
    "MixedSingleTaskGP": ModelRegistryEntry(
        "MixedSingleTaskGP",
        ModelCapabilities(input_type=InputType.MIXED),
        _docs(
            "docs/models/standard.md",
            "docs/theory/05_mixed_variables.md",
            "examples/notebooks/02_mixed_single_task_gp.ipynb",
        ),
        "native mixed-variable GP path",
    ),
    "KroneckerMultiTaskGP": ModelRegistryEntry(
        "KroneckerMultiTaskGP",
        ModelCapabilities(task_type=TaskType.MULTITASK),
        _docs(
            "docs/models/multitask_multioutput.md",
            "docs/theory/07_multitask_multioutput.md",
            "examples/notebooks/04_multitask_gp.ipynb",
        ),
        "Kronecker-structured exact multitask GP",
    ),
    "MixedSaasFullyBayesianMultiTaskGP": ModelRegistryEntry(
        "MixedSaasFullyBayesianMultiTaskGP",
        ModelCapabilities(
            input_type=InputType.MIXED,
            task_type=TaskType.MULTITASK,
            inference=InferenceType.FULLY_BAYESIAN,
            high_dimensional=HighDimensionalStrategy.SAAS,
        ),
        _docs(
            "docs/models/multitask_multioutput.md",
            "docs/theory/21_advanced_high_dimensional_models.md",
            "examples/notebooks/08_saas_gp.ipynb",
        ),
        "fully Bayesian SAAS multitask model with mixed inputs",
    ),
    "PCAGP": ModelRegistryEntry(
        "PCAGP",
        ModelCapabilities(high_dimensional=HighDimensionalStrategy.REDUCTION),
        _docs(
            "docs/models/reduced.md",
            "docs/theory/18_dimensionality_reduction_gp.md",
            "examples/notebooks/19_reduced_gp.ipynb",
        ),
        "PCA reduction followed by an exact GP",
    ),
    "MixedRobustRelevancePursuitMultiTaskGP": ModelRegistryEntry(
        "MixedRobustRelevancePursuitMultiTaskGP",
        ModelCapabilities(
            input_type=InputType.MIXED,
            task_type=TaskType.MULTITASK,
            robustness=frozenset({RobustnessType.RELEVANCE_PURSUIT}),
        ),
        _docs(
            "docs/models/robust_noise.md",
            "docs/theory/14_robust_gaussian_process.md",
            "examples/notebooks/10_robust_gp.ipynb",
        ),
        "robust relevance-pursuit multitask GP with mixed inputs",
    ),
    "SingleTaskMultiFidelityGP": ModelRegistryEntry(
        "SingleTaskMultiFidelityGP",
        ModelCapabilities(multi_fidelity=True),
        _docs(
            "docs/models/standard.md",
            "docs/theory/06_multi_fidelity.md",
            "examples/notebooks/03_multi_fidelity_gp.ipynb",
        ),
        "BoTorch single-task multi-fidelity GP wrapper",
    ),
    "SingleTaskVariationalGP": ModelRegistryEntry(
        "SingleTaskVariationalGP",
        ModelCapabilities(inference=InferenceType.VARIATIONAL),
        _docs(
            "docs/models/standard.md",
            "docs/theory/09_variational_gp.md",
            "examples/notebooks/06_variational_gp.ipynb",
        ),
        "variational single-task GP",
    ),
}


# Phase 3 expands the registry to every public model family. Metadata here is
# explicit and reviewed; helpers only remove repetition inside a verified family.
def _register_family(
    names: tuple[str, ...],
    *,
    guide: str,
    theory: str,
    notebook: str,
    strategy: str,
    high_dimensional: HighDimensionalStrategy = HighDimensionalStrategy.NONE,
    robustness: frozenset[RobustnessType] = frozenset(),
    multi_fidelity: bool = False,
    structured_output: bool = False,
    preference: bool = False,
    non_gp: bool = False,
) -> None:
    for name in names:
        MODEL_REGISTRY.setdefault(
            name,
            ModelRegistryEntry(
                name,
                ModelCapabilities(
                    input_type=(
                        InputType.MIXED
                        if name.startswith("Mixed")
                        else InputType.CONTINUOUS
                    ),
                    task_type=TaskType.MULTITASK
                    if (
                        "MultiTask" in name
                        or name in {"HeterogeneousMTGP", "MixedHeterogeneousMTGP"}
                    )
                    else TaskType.SINGLE,
                    inference=InferenceType.FULLY_BAYESIAN
                    if "FullyBayesian" in name
                    else InferenceType.VARIATIONAL
                    if "Variational" in name
                    else InferenceType.EXACT,
                    high_dimensional=high_dimensional,
                    robustness=robustness,
                    multi_fidelity=multi_fidelity,
                    structured_output=structured_output,
                    preference=preference,
                    non_gp=non_gp,
                ),
                _docs(guide, theory, notebook),
                strategy,
            ),
        )


_register_family(
    ("MixedSingleTaskMultiFidelityGP",),
    guide="docs/models/standard.md",
    theory="docs/theory/06_multi_fidelity.md",
    notebook="examples/notebooks/03_multi_fidelity_gp.ipynb",
    strategy="mixed multi-fidelity GP", multi_fidelity=True,
)
_register_family(
    ("MixedSingleTaskVariationalGP",),
    guide="docs/models/standard.md",
    theory="docs/theory/09_variational_gp.md",
    notebook="examples/notebooks/06_variational_gp.ipynb",
    strategy="mixed variational GP",
)
_register_family(
    ("MultiTaskGP", "MixedMultiTaskGP", "MixedKroneckerMultiTaskGP", "ModelListGP"),
    guide="docs/models/multitask_multioutput.md",
    theory="docs/theory/07_multitask_multioutput.md",
    notebook="examples/notebooks/04_multitask_gp.ipynb",
    strategy="multitask or multi-output GP",
)
_register_family(
    ("SaasFullyBayesianSingleTaskGP", "SaasFullyBayesianMultiTaskGP",
     "MixedSaasFullyBayesianSingleTaskGP"),
    guide="docs/models/high_dimensional.md",
    theory="docs/theory/21_advanced_high_dimensional_models.md",
    notebook="examples/notebooks/08_saas_gp.ipynb",
    strategy="fully Bayesian SAAS GP",
    high_dimensional=HighDimensionalStrategy.SAAS,
)
_register_family(
    ("AdditiveMapSaasSingleTaskGP", "EnsembleMapSaasSingleTaskGP",
     "MixedAdditiveMapSaasSingleTaskGP", "MixedEnsembleMapSaasSingleTaskGP"),
    guide="docs/models/high_dimensional.md",
    theory="docs/theory/21_advanced_high_dimensional_models.md",
    notebook="examples/notebooks/09_map_saas_and_additive_gp.ipynb",
    strategy="MAP-SAAS GP",
    high_dimensional=HighDimensionalStrategy.MAP_SAAS,
)
_register_family(
    ("ALEBOGP",), guide="docs/models/high_dimensional.md",
    theory="docs/theory/21_advanced_high_dimensional_models.md",
    notebook="examples/notebooks/09_map_saas_and_additive_gp.ipynb",
    strategy="ALEBO random embedding",
    high_dimensional=HighDimensionalStrategy.RANDOM_EMBEDDING,
)
_register_family(
    ("OrthogonalAdditiveGP", "MixedOrthogonalAdditiveGP"),
    guide="docs/models/high_dimensional.md",
    theory="docs/theory/21_advanced_high_dimensional_models.md",
    notebook="examples/notebooks/09_map_saas_and_additive_gp.ipynb",
    strategy="orthogonal additive GP",
    high_dimensional=HighDimensionalStrategy.ADDITIVE,
)

_REDUCED = (
    "ReducedGP",
    "PCAGP",
    "PLSGP",
    "RandomProjectionGP",
    "OutputPCAGP",
    "OutputPLSGP",
    "MixedReducedGP", "MixedPCAGP", "MixedPLSGP", "MixedRandomProjectionGP",
    "ReducedMultiTaskGP", "ReducedKroneckerMultiTaskGP", "PCAMultiTaskGP",
    "PCAKroneckerMultiTaskGP", "PLSMultiTaskGP", "PLSKroneckerMultiTaskGP",
    "RandomProjectionMultiTaskGP", "RandomProjectionKroneckerMultiTaskGP",
    "MixedReducedMultiTaskGP", "MixedReducedKroneckerMultiTaskGP",
)
_register_family(
    _REDUCED, guide="docs/models/reduced.md",
    theory="docs/theory/18_dimensionality_reduction_gp.md",
    notebook="examples/notebooks/19_reduced_gp.ipynb",
    strategy="explicit dimensionality reduction GP",
    high_dimensional=HighDimensionalStrategy.REDUCTION,
)
_NEURAL_REDUCED = (
    "AutoEncoderGP",
    "VAEGP",
    "SupervisedAutoEncoderGP",
    "SupervisedVAEGP",
    "HybridAutoEncoderGP",
    "JointEncoderGP", "JointVAEGP", "MixedAutoEncoderGP", "MixedVAEGP",
    "MixedSupervisedAutoEncoderGP", "MixedSupervisedVAEGP", "MixedHybridAutoEncoderGP",
    "MixedJointEncoderGP", "MixedJointVAEGP", "AutoEncoderMultiTaskGP",
    "AutoEncoderKroneckerMultiTaskGP", "VAEKroneckerMultiTaskGP", "VAEMultiTaskGP",
    "SupervisedAutoEncoderMultiTaskGP", "SupervisedAutoEncoderKroneckerMultiTaskGP",
    "SupervisedVAEMultiTaskGP", "SupervisedVAEKroneckerMultiTaskGP", "HybridAutoEncoderMultiTaskGP",
    "HybridAutoEncoderKroneckerMultiTaskGP", "JointEncoderMultiTaskGP",
    "JointEncoderKroneckerMultiTaskGP", "JointVAEMultiTaskGP", "JointVAEKroneckerMultiTaskGP",
    "MixedJointEncoderMultiTaskGP",
)
_register_family(
    _NEURAL_REDUCED, guide="docs/models/reduced.md",
    theory="docs/theory/19_neural_reduction_gp.md",
    notebook="examples/notebooks/20_neural_reduction_gp.ipynb",
    strategy="neural dimensionality reduction GP",
    high_dimensional=HighDimensionalStrategy.NEURAL_REDUCTION,
)

_register_family(
    ("RobustRelevancePursuitSingleTaskGP", "RobustRelevancePursuitMultiTaskGP",
     "MixedRobustRelevancePursuitSingleTaskGP"),
    guide="docs/models/robust_noise.md",
    theory="docs/theory/14_robust_gaussian_process.md",
    notebook="examples/notebooks/10_robust_gp.ipynb",
    strategy="robust relevance-pursuit GP",
    robustness=frozenset({RobustnessType.RELEVANCE_PURSUIT}),
)
_register_family(
    ("ContaminatedSingleTaskGP", "ContaminatedMultiTaskGP", "MixedContaminatedSingleTaskGP",
     "MixedContaminatedMultiTaskGP"),
    guide="docs/models/robust_noise.md",
    theory="docs/theory/14_robust_gaussian_process.md",
    notebook="examples/notebooks/15_robust_observation_models.ipynb",
    strategy="contaminated-observation GP",
    robustness=frozenset({RobustnessType.CONTAMINATION}),
)
_register_family(
    ("StudentTSingleTaskGP", "StudentTMultiTaskGP", "MixedStudentTSingleTaskGP",
     "MixedStudentTMultiTaskGP"),
    guide="docs/models/robust_noise.md",
    theory="docs/theory/14_robust_gaussian_process.md",
    notebook="examples/notebooks/15_robust_observation_models.ipynb",
    strategy="Student-t robust GP",
    robustness=frozenset({RobustnessType.STUDENT_T}),
)
_register_family(
    (
        "HeteroskedasticSingleTaskGP",
        "HeteroskedasticMultiTaskGP",
        "MixedHeteroskedasticSingleTaskGP",
     "MixedHeteroskedasticMultiTaskGP", "JointHeteroskedasticSingleTaskGP",
     "MixedJointHeteroskedasticSingleTaskGP", "ReplicateNoiseSingleTaskGP",
     "MixedReplicateNoiseSingleTaskGP"),
    guide="docs/models/robust_noise.md",
    theory="docs/theory/15_heteroskedastic_noise.md",
    notebook="examples/notebooks/16_noise_models.ipynb",
    strategy="heteroskedastic or replicate-noise GP",
    robustness=frozenset({RobustnessType.HETEROSKEDASTIC}),
)
_register_family(
    ("NonstationarySingleTaskGP", "NonstationaryMultiTaskGP", "MixedNonstationarySingleTaskGP",
     "MixedNonstationaryMultiTaskGP"),
    guide="docs/models/robust_noise.md",
    theory="docs/theory/17_nonstationary_gp.md",
    notebook="examples/notebooks/18_nonstationary_gp.ipynb",
    strategy="nonstationary GP",
    robustness=frozenset({RobustnessType.NONSTATIONARY}),
)
_register_family(
    (
        "UncertainInputSingleTaskGP",
        "MixedUncertainInputSingleTaskGP",
        "UncertainCategoricalSingleTaskGP",
    ),
    guide="docs/models/uncertain_input.md",
    theory="docs/theory/16_uncertain_input_gp.md",
    notebook="examples/notebooks/17_uncertain_input_gp.ipynb",
    strategy="uncertain-input GP",
    robustness=frozenset({RobustnessType.UNCERTAIN_INPUT}),
)

_register_family(
    ("HigherOrderGP", "MixedHigherOrderGP", "LatentKroneckerGP", "MixedLatentKroneckerGP"),
    guide="docs/models/structured_output.md",
    theory="docs/theory/11_structured_output.md",
    notebook="examples/notebooks/11_structured_output_gp.ipynb",
    strategy="structured-output GP",
    structured_output=True,
)
_register_family(
    ("HeterogeneousMTGP", "MixedHeterogeneousMTGP", "HierarchicalConditionalKernelMultiTaskGP",
     "MixedHierarchicalConditionalKernelMultiTaskGP"),
    guide="docs/models/multitask_multioutput.md",
    theory="docs/theory/07_multitask_multioutput.md",
    notebook="examples/notebooks/13_heterogeneous_multitask_gp.ipynb",
    strategy="structured multitask GP",
)
_register_family(
    (
        "HierarchicalConditionalKernelGP",
        "MixedHierarchicalConditionalKernelGP",
        "LCEAGP",
        "LCEMGP",
     "MixedLCEMGP", "SACGP"),
    guide="docs/models/hierarchical_contextual.md",
    theory="docs/theory/12_hierarchical_contextual_gp.md",
    notebook="examples/notebooks/14_contextual_gp.ipynb",
    strategy="hierarchical or contextual GP",
)
_register_family(
    ("PairwiseGP",),
    guide="docs/models/preference.md",
    theory="docs/theory/10_preference_learning.md",
    notebook="examples/notebooks/07_pairwise_gp.ipynb",
    strategy="pairwise preference GP", preference=True,
)

_register_family(
    ("SingleTaskDeepGP", "MultiTaskDeepGP", "MixedSingleTaskDeepGP", "MixedMultiTaskDeepGP"),
    guide="docs/models/expressive.md",
    theory="docs/theory/22_expressive_gp.md",
    notebook="examples/notebooks/24_expressive_surrogate_gp.ipynb",
    strategy="deep GP",
    high_dimensional=HighDimensionalStrategy.DEEP,
)
_register_family(
    ("InfiniteWidthBNNGP", "InfiniteWidthBNNMultiTaskGP", "MixedInfiniteWidthBNNGP",
     "MixedInfiniteWidthBNNMultiTaskGP", "SpectralMixtureGP", "SpectralMixtureMultiTaskGP",
     "MixedSpectralMixtureGP", "MixedSpectralMixtureMultiTaskGP"),
    guide="docs/models/expressive.md",
    theory="docs/theory/22_expressive_gp.md",
    notebook="examples/notebooks/24_expressive_surrogate_gp.ipynb",
    strategy="expressive GP surrogate",
)
_register_family(
    ("RandomForestSurrogate", "ExtraTreesSurrogate", "GradientBoostingSurrogate",
     "HistGradientBoostingSurrogate"),
    guide="docs/models/non_gp.md",
    theory="docs/theory/23_non_gp_surrogates.md",
    notebook="examples/notebooks/25_non_gp_surrogates.ipynb",
    strategy="non-GP tree ensemble surrogate",
    non_gp=True,
)


def get_model_registry_entry(model_name: str) -> ModelRegistryEntry:
    """Return registry metadata for a registered public model."""
    return MODEL_REGISTRY[model_name]
