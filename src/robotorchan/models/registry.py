"""Initial model capability registry."""

from robotorchan.models.capabilities import (
    DocumentationLinks,
    HighDimensionalStrategy,
    InferenceType,
    InputType,
    ModelCapabilities,
    ModelRegistryEntry,
    PosteriorSamplingType,
    RobustnessType,
    TaskType,
)


def _docs(guide: str, theory: str, notebook: str) -> DocumentationLinks:
    return DocumentationLinks(guide, theory, notebook)


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "SingleTaskGP": ModelRegistryEntry(
        "SingleTaskGP",
        ModelCapabilities(
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
        _docs(
            "docs/models/standard.md",
            "docs/theory/02_gaussian_process.md",
            "examples/notebooks/01_single_task_gp.ipynb",
        ),
        "BoTorch exact single-task GP wrapper",
    ),
    "MixedSingleTaskGP": ModelRegistryEntry(
        "MixedSingleTaskGP",
        ModelCapabilities(
            input_type=InputType.MIXED,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
        _docs(
            "docs/models/standard.md",
            "docs/theory/05_mixed_variables.md",
            "examples/notebooks/02_mixed_single_task_gp.ipynb",
        ),
        "native mixed-variable GP path",
    ),
    "KroneckerMultiTaskGP": ModelRegistryEntry(
        "KroneckerMultiTaskGP",
        ModelCapabilities(
            task_type=TaskType.MULTITASK,
            supports_multi_output=True,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
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
            supports_multi_output=True,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
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
        ModelCapabilities(
            high_dimensional=HighDimensionalStrategy.REDUCTION,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
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
            supports_multi_output=True,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
        _docs(
            "docs/models/robust_noise.md",
            "docs/theory/14_robust_gaussian_process.md",
            "examples/notebooks/10_robust_gp.ipynb",
        ),
        "robust relevance-pursuit multitask GP with mixed inputs",
    ),
    "PCAMultiFidelityGP": ModelRegistryEntry(
        "PCAMultiFidelityGP",
        ModelCapabilities(
            high_dimensional=HighDimensionalStrategy.REDUCTION,
            multi_fidelity=True,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
        _docs(
            "docs/models/reduced.md",
            "docs/theory/06_multi_fidelity.md",
            "examples/notebooks/03_multi_fidelity_gp.ipynb",
        ),
        "PCA design reduction with BoTorch multi-fidelity covariance",
    ),
    "HeteroskedasticMultiFidelityGP": ModelRegistryEntry(
        "HeteroskedasticMultiFidelityGP",
        ModelCapabilities(
            robustness=frozenset({RobustnessType.HETEROSKEDASTIC}),
            multi_fidelity=True,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
        _docs(
            "docs/models/robust_noise.md",
            "docs/theory/15_heteroskedastic_noise.md",
            "examples/notebooks/16_noise_models.ipynb",
        ),
        "iterative multi-fidelity response and log-noise GPs",
    ),
    "ReplicateNoiseMultiFidelityGP": ModelRegistryEntry(
        "ReplicateNoiseMultiFidelityGP",
        ModelCapabilities(
            robustness=frozenset({RobustnessType.REPLICATE_NOISE}),
            multi_fidelity=True,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
        _docs(
            "docs/models/robust_noise.md",
            "docs/theory/15_heteroskedastic_noise.md",
            "examples/notebooks/16_noise_models.ipynb",
        ),
        "multi-fidelity GP with empirical replicate observation noise",
    ),
    "SingleTaskMultiFidelityGP": ModelRegistryEntry(
        "SingleTaskMultiFidelityGP",
        ModelCapabilities(
            multi_fidelity=True,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
        _docs(
            "docs/models/standard.md",
            "docs/theory/06_multi_fidelity.md",
            "examples/notebooks/03_multi_fidelity_gp.ipynb",
        ),
        "BoTorch single-task multi-fidelity GP wrapper",
    ),
    "SingleTaskVariationalGP": ModelRegistryEntry(
        "SingleTaskVariationalGP",
        ModelCapabilities(
            inference=InferenceType.VARIATIONAL,
            supports_posterior_samples=True,
            posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
            supports_fantasize=True,
        ),
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
# Capability groups are explicit metadata. Do not derive them from class names.
_MIXED_MODELS = frozenset(
    {
        "MixedSingleTaskMultiFidelityGP",
        "MixedSingleTaskVariationalGP",
        "MixedMultiTaskGP",
        "MixedKroneckerMultiTaskGP",
        "MixedSaasFullyBayesianSingleTaskGP",
        "MixedAdditiveMapSaasSingleTaskGP",
        "MixedEnsembleMapSaasSingleTaskGP",
        "MixedOrthogonalAdditiveGP",
        "MixedReducedGP",
        "MixedPCAGP",
        "MixedPLSGP",
        "MixedRandomProjectionGP",
        "MixedReducedMultiTaskGP",
        "MixedReducedKroneckerMultiTaskGP",
        "MixedAutoEncoderGP",
        "MixedVAEGP",
        "MixedSupervisedAutoEncoderGP",
        "MixedSupervisedVAEGP",
        "MixedHybridAutoEncoderGP",
        "MixedJointEncoderGP",
        "MixedJointVAEGP",
        "MixedJointEncoderMultiTaskGP",
        "MixedRobustRelevancePursuitSingleTaskGP",
        "MixedContaminatedSingleTaskGP",
        "MixedContaminatedMultiTaskGP",
        "MixedStudentTSingleTaskGP",
        "MixedStudentTMultiTaskGP",
        "MixedHeteroskedasticSingleTaskGP",
        "MixedHeteroskedasticMultiTaskGP",
        "MixedJointHeteroskedasticSingleTaskGP",
        "MixedReplicateNoiseSingleTaskGP",
        "MixedNonstationarySingleTaskGP",
        "MixedNonstationaryMultiTaskGP",
        "MixedUncertainInputSingleTaskGP",
        "MixedHigherOrderGP",
        "MixedLatentKroneckerGP",
        "MixedHeterogeneousMTGP",
        "MixedHierarchicalConditionalKernelMultiTaskGP",
        "MixedHierarchicalConditionalKernelGP",
        "MixedLCEMGP",
        "MixedSingleTaskDeepGP",
        "MixedMultiTaskDeepGP",
        "MixedInfiniteWidthBNNGP",
        "MixedInfiniteWidthBNNMultiTaskGP",
        "MixedSpectralMixtureGP",
        "MixedSpectralMixtureMultiTaskGP",
    }
)
_MULTITASK_MODELS = frozenset(
    {
        "MultiTaskGP",
        "MixedMultiTaskGP",
        "MixedKroneckerMultiTaskGP",
        "SaasFullyBayesianMultiTaskGP",
        "ReducedMultiTaskGP",
        "ReducedKroneckerMultiTaskGP",
        "PCAMultiTaskGP",
        "PCAKroneckerMultiTaskGP",
        "PLSMultiTaskGP",
        "PLSKroneckerMultiTaskGP",
        "RandomProjectionMultiTaskGP",
        "RandomProjectionKroneckerMultiTaskGP",
        "MixedReducedMultiTaskGP",
        "MixedReducedKroneckerMultiTaskGP",
        "AutoEncoderMultiTaskGP",
        "AutoEncoderKroneckerMultiTaskGP",
        "VAEKroneckerMultiTaskGP",
        "VAEMultiTaskGP",
        "SupervisedAutoEncoderMultiTaskGP",
        "SupervisedAutoEncoderKroneckerMultiTaskGP",
        "SupervisedVAEMultiTaskGP",
        "SupervisedVAEKroneckerMultiTaskGP",
        "HybridAutoEncoderMultiTaskGP",
        "HybridAutoEncoderKroneckerMultiTaskGP",
        "JointEncoderMultiTaskGP",
        "JointEncoderKroneckerMultiTaskGP",
        "JointVAEMultiTaskGP",
        "JointVAEKroneckerMultiTaskGP",
        "MixedJointEncoderMultiTaskGP",
        "RobustRelevancePursuitMultiTaskGP",
        "ContaminatedMultiTaskGP",
        "MixedContaminatedMultiTaskGP",
        "StudentTMultiTaskGP",
        "MixedStudentTMultiTaskGP",
        "HeteroskedasticMultiTaskGP",
        "MixedHeteroskedasticMultiTaskGP",
        "NonstationaryMultiTaskGP",
        "MixedNonstationaryMultiTaskGP",
        "HeterogeneousMTGP",
        "MixedHeterogeneousMTGP",
        "HierarchicalConditionalKernelMultiTaskGP",
        "MixedHierarchicalConditionalKernelMultiTaskGP",
        "LCEMGP",
        "MixedLCEMGP",
        "MultiTaskDeepGP",
        "MixedMultiTaskDeepGP",
        "InfiniteWidthBNNMultiTaskGP",
        "MixedInfiniteWidthBNNMultiTaskGP",
        "SpectralMixtureMultiTaskGP",
        "MixedSpectralMixtureMultiTaskGP",
    }
)
_VARIATIONAL_MODELS = frozenset(
    {
        "MixedSingleTaskVariationalGP",
        "SingleTaskDeepGP",
        "MultiTaskDeepGP",
        "MixedSingleTaskDeepGP",
        "MixedMultiTaskDeepGP",
        "ContaminatedSingleTaskGP",
        "ContaminatedMultiTaskGP",
        "MixedContaminatedSingleTaskGP",
        "MixedContaminatedMultiTaskGP",
        "StudentTSingleTaskGP",
        "StudentTMultiTaskGP",
        "MixedStudentTSingleTaskGP",
        "MixedStudentTMultiTaskGP",
        "JointHeteroskedasticSingleTaskGP",
        "MixedJointHeteroskedasticSingleTaskGP",
    }
)
_FULLY_BAYESIAN_MODELS = frozenset(
    {
        "SaasFullyBayesianSingleTaskGP",
        "SaasFullyBayesianMultiTaskGP",
        "MixedSaasFullyBayesianSingleTaskGP",
    }
)
_MULTI_OUTPUT_MODELS = frozenset(
    {
        "AutoEncoderKroneckerMultiTaskGP",
        "AutoEncoderMultiTaskGP",
        "ContaminatedMultiTaskGP",
        "HeterogeneousMTGP",
        "HeteroskedasticMultiTaskGP",
        "HierarchicalConditionalKernelMultiTaskGP",
        "HybridAutoEncoderKroneckerMultiTaskGP",
        "HybridAutoEncoderMultiTaskGP",
        "InfiniteWidthBNNMultiTaskGP",
        "JointEncoderKroneckerMultiTaskGP",
        "JointEncoderMultiTaskGP",
        "JointVAEKroneckerMultiTaskGP",
        "JointVAEMultiTaskGP",
        "KroneckerMultiTaskGP",
        "LCEMGP",
        "MultiTaskDeepGP",
        "MultiTaskGP",
        "MixedContaminatedMultiTaskGP",
        "MixedHeterogeneousMTGP",
        "MixedHeteroskedasticMultiTaskGP",
        "MixedHierarchicalConditionalKernelMultiTaskGP",
        "MixedInfiniteWidthBNNMultiTaskGP",
        "MixedJointEncoderMultiTaskGP",
        "MixedKroneckerMultiTaskGP",
        "MixedLCEMGP",
        "MixedMultiTaskGP",
        "MixedMultiTaskDeepGP",
        "MixedNonstationaryMultiTaskGP",
        "MixedReducedKroneckerMultiTaskGP",
        "MixedReducedMultiTaskGP",
        "MixedRobustRelevancePursuitMultiTaskGP",
        "MixedSaasFullyBayesianMultiTaskGP",
        "MixedSpectralMixtureMultiTaskGP",
        "MixedStudentTMultiTaskGP",
        "NonstationaryMultiTaskGP",
        "PCAKroneckerMultiTaskGP",
        "PCAMultiTaskGP",
        "PLSKroneckerMultiTaskGP",
        "PLSMultiTaskGP",
        "RandomProjectionKroneckerMultiTaskGP",
        "RandomProjectionMultiTaskGP",
        "ReducedKroneckerMultiTaskGP",
        "ReducedMultiTaskGP",
        "RobustRelevancePursuitMultiTaskGP",
        "SaasFullyBayesianMultiTaskGP",
        "SpectralMixtureMultiTaskGP",
        "StudentTMultiTaskGP",
        "SupervisedAutoEncoderKroneckerMultiTaskGP",
        "SupervisedAutoEncoderMultiTaskGP",
        "SupervisedVAEKroneckerMultiTaskGP",
        "SupervisedVAEMultiTaskGP",
        "VAEKroneckerMultiTaskGP",
        "VAEMultiTaskGP",
        "ModelListGP",
    }
)


_POSTERIOR_SAMPLING_MODELS = frozenset(
    {
        name
        for name in MODEL_REGISTRY
        if MODEL_REGISTRY[name].capabilities.supports_posterior_samples
    }
    | {
        "MixedSingleTaskMultiFidelityGP",
        "MixedSingleTaskVariationalGP",
        "MultiTaskGP",
        "MixedMultiTaskGP",
        "MixedKroneckerMultiTaskGP",
        "ModelListGP",
        "RandomForestSurrogate",
        "ExtraTreesSurrogate",
        "GradientBoostingSurrogate",
        "HistGradientBoostingSurrogate",
        "NGBoostSurrogate",
        "EnsembleMapSaasSingleTaskGP",
        "MixedEnsembleMapSaasSingleTaskGP",
        "MixedPCAGP",
        "MixedPLSGP",
        "MixedRandomProjectionGP",
        "MixedReducedGP",
        "ReducedGP",
        "PCAGP",
        "PLSGP",
        "RandomProjectionGP",
    }
)


_FANTASIZE_MODELS = frozenset(
    {name for name in MODEL_REGISTRY if MODEL_REGISTRY[name].capabilities.supports_fantasize}
    | {
        "MixedSingleTaskMultiFidelityGP",
        "MultiTaskGP",
        "MixedMultiTaskGP",
        "MixedKroneckerMultiTaskGP",
        "ModelListGP",
        "MixedReducedGP",
        "MixedPCAGP",
        "MixedPLSGP",
        "MixedRandomProjectionGP",
        "ReducedGP",
        "PCAGP",
        "PLSGP",
        "RandomProjectionGP",
    }
)


_GAUSSIAN_ENSEMBLE_MODELS = frozenset(
    {
        "EnsembleMapSaasSingleTaskGP",
        "MixedEnsembleMapSaasSingleTaskGP",
    }
)


_ENSEMBLE_POSTERIOR_MODELS = frozenset(
    {
        "RandomForestSurrogate",
        "ExtraTreesSurrogate",
        "GradientBoostingSurrogate",
        "HistGradientBoostingSurrogate",
    }
)


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
        inference = InferenceType.EXACT
        if name in _VARIATIONAL_MODELS:
            inference = InferenceType.VARIATIONAL
        elif name in _FULLY_BAYESIAN_MODELS:
            inference = InferenceType.FULLY_BAYESIAN
        elif non_gp:
            inference = InferenceType.NOT_APPLICABLE
        MODEL_REGISTRY.setdefault(
            name,
            ModelRegistryEntry(
                name,
                ModelCapabilities(
                    input_type=InputType.MIXED if name in _MIXED_MODELS else InputType.CONTINUOUS,
                    task_type=TaskType.MULTITASK if name in _MULTITASK_MODELS else TaskType.SINGLE,
                    inference=inference,
                    high_dimensional=high_dimensional,
                    robustness=robustness,
                    multi_fidelity=multi_fidelity,
                    structured_output=structured_output,
                    preference=preference,
                    non_gp=non_gp,
                    ensemble_posterior=name
                    in (_GAUSSIAN_ENSEMBLE_MODELS | _ENSEMBLE_POSTERIOR_MODELS),
                    supports_multi_output=name in _MULTI_OUTPUT_MODELS,
                    supports_posterior_samples=name in _POSTERIOR_SAMPLING_MODELS,
                    posterior_sampling_type=(
                        PosteriorSamplingType.ENSEMBLE
                        if name in _ENSEMBLE_POSTERIOR_MODELS
                        else (
                            PosteriorSamplingType.GAUSSIAN
                            if name in _POSTERIOR_SAMPLING_MODELS
                            else PosteriorSamplingType.NONE
                        )
                    ),
                    supports_fantasize=name in _FANTASIZE_MODELS,
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
    strategy="mixed multi-fidelity GP",
    multi_fidelity=True,
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
    (
        "SaasFullyBayesianSingleTaskGP",
        "SaasFullyBayesianMultiTaskGP",
        "MixedSaasFullyBayesianSingleTaskGP",
    ),
    guide="docs/models/high_dimensional.md",
    theory="docs/theory/21_advanced_high_dimensional_models.md",
    notebook="examples/notebooks/08_saas_gp.ipynb",
    strategy="fully Bayesian SAAS GP",
    high_dimensional=HighDimensionalStrategy.SAAS,
)
_register_family(
    (
        "AdditiveMapSaasSingleTaskGP",
        "EnsembleMapSaasSingleTaskGP",
        "MixedAdditiveMapSaasSingleTaskGP",
        "MixedEnsembleMapSaasSingleTaskGP",
    ),
    guide="docs/models/high_dimensional.md",
    theory="docs/theory/21_advanced_high_dimensional_models.md",
    notebook="examples/notebooks/09_map_saas_and_additive_gp.ipynb",
    strategy="MAP-SAAS GP",
    high_dimensional=HighDimensionalStrategy.MAP_SAAS,
)
_register_family(
    ("ALEBOGP",),
    guide="docs/models/high_dimensional.md",
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
    "MixedReducedGP",
    "MixedPCAGP",
    "MixedPLSGP",
    "MixedRandomProjectionGP",
    "ReducedMultiTaskGP",
    "ReducedKroneckerMultiTaskGP",
    "PCAMultiTaskGP",
    "PCAKroneckerMultiTaskGP",
    "PLSMultiTaskGP",
    "PLSKroneckerMultiTaskGP",
    "RandomProjectionMultiTaskGP",
    "RandomProjectionKroneckerMultiTaskGP",
    "MixedReducedMultiTaskGP",
    "MixedReducedKroneckerMultiTaskGP",
)
_register_family(
    _REDUCED,
    guide="docs/models/reduced.md",
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
    "JointEncoderGP",
    "JointVAEGP",
    "MixedAutoEncoderGP",
    "MixedVAEGP",
    "MixedSupervisedAutoEncoderGP",
    "MixedSupervisedVAEGP",
    "MixedHybridAutoEncoderGP",
    "MixedJointEncoderGP",
    "MixedJointVAEGP",
    "AutoEncoderMultiTaskGP",
    "AutoEncoderKroneckerMultiTaskGP",
    "VAEKroneckerMultiTaskGP",
    "VAEMultiTaskGP",
    "SupervisedAutoEncoderMultiTaskGP",
    "SupervisedAutoEncoderKroneckerMultiTaskGP",
    "SupervisedVAEMultiTaskGP",
    "SupervisedVAEKroneckerMultiTaskGP",
    "HybridAutoEncoderMultiTaskGP",
    "HybridAutoEncoderKroneckerMultiTaskGP",
    "JointEncoderMultiTaskGP",
    "JointEncoderKroneckerMultiTaskGP",
    "JointVAEMultiTaskGP",
    "JointVAEKroneckerMultiTaskGP",
    "MixedJointEncoderMultiTaskGP",
)
_register_family(
    _NEURAL_REDUCED,
    guide="docs/models/reduced.md",
    theory="docs/theory/19_neural_reduction_gp.md",
    notebook="examples/notebooks/20_neural_reduction_gp.ipynb",
    strategy="neural dimensionality reduction GP",
    high_dimensional=HighDimensionalStrategy.NEURAL_REDUCTION,
)

_register_family(
    (
        "RobustRelevancePursuitSingleTaskGP",
        "RobustRelevancePursuitMultiTaskGP",
        "MixedRobustRelevancePursuitSingleTaskGP",
    ),
    guide="docs/models/robust_noise.md",
    theory="docs/theory/14_robust_gaussian_process.md",
    notebook="examples/notebooks/10_robust_gp.ipynb",
    strategy="robust relevance-pursuit GP",
    robustness=frozenset({RobustnessType.RELEVANCE_PURSUIT}),
)
_register_family(
    (
        "ContaminatedSingleTaskGP",
        "ContaminatedMultiTaskGP",
        "MixedContaminatedSingleTaskGP",
        "MixedContaminatedMultiTaskGP",
    ),
    guide="docs/models/robust_noise.md",
    theory="docs/theory/14_robust_gaussian_process.md",
    notebook="examples/notebooks/15_robust_observation_models.ipynb",
    strategy="contaminated-observation GP",
    robustness=frozenset({RobustnessType.CONTAMINATION}),
)
_register_family(
    (
        "StudentTSingleTaskGP",
        "StudentTMultiTaskGP",
        "MixedStudentTSingleTaskGP",
        "MixedStudentTMultiTaskGP",
    ),
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
        "MixedHeteroskedasticMultiTaskGP",
        "JointHeteroskedasticSingleTaskGP",
        "MixedJointHeteroskedasticSingleTaskGP",
        "ReplicateNoiseSingleTaskGP",
        "MixedReplicateNoiseSingleTaskGP",
    ),
    guide="docs/models/robust_noise.md",
    theory="docs/theory/15_heteroskedastic_noise.md",
    notebook="examples/notebooks/16_noise_models.ipynb",
    strategy="heteroskedastic or replicate-noise GP",
    robustness=frozenset({RobustnessType.HETEROSKEDASTIC}),
)
_register_family(
    (
        "NonstationarySingleTaskGP",
        "NonstationaryMultiTaskGP",
        "NonstationaryKroneckerMultiTaskGP",
        "NonstationaryKroneckerMultiTaskGP",
        "MixedNonstationarySingleTaskGP",
        "MixedNonstationaryMultiTaskGP",
    ),
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
    (
        "HigherOrderGP",
        "MixedHigherOrderGP",
        "LatentKroneckerGP",
        "MixedLatentKroneckerGP",
    ),
    guide="docs/models/structured_output.md",
    theory="docs/theory/11_structured_output.md",
    notebook="examples/notebooks/11_structured_output_gp.ipynb",
    strategy="structured-output GP",
    structured_output=True,
)
_register_family(
    (
        "HeterogeneousMTGP",
        "MixedHeterogeneousMTGP",
        "HierarchicalConditionalKernelMultiTaskGP",
        "MixedHierarchicalConditionalKernelMultiTaskGP",
    ),
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
        "MixedLCEMGP",
        "SACGP",
    ),
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
    strategy="pairwise preference GP",
    preference=True,
)

_register_family(
    (
        "SingleTaskDeepGP",
        "MultiTaskDeepGP",
        "MixedSingleTaskDeepGP",
        "MixedMultiTaskDeepGP",
    ),
    guide="docs/models/expressive.md",
    theory="docs/theory/22_expressive_gp.md",
    notebook="examples/notebooks/24_expressive_surrogate_gp.ipynb",
    strategy="deep GP",
    high_dimensional=HighDimensionalStrategy.DEEP,
)
_register_family(
    (
        "InfiniteWidthBNNGP",
        "InfiniteWidthBNNMultiTaskGP",
        "MixedInfiniteWidthBNNGP",
        "MixedInfiniteWidthBNNMultiTaskGP",
        "SpectralMixtureGP",
        "SpectralMixtureMultiTaskGP",
        "MixedSpectralMixtureGP",
        "MixedSpectralMixtureMultiTaskGP",
    ),
    guide="docs/models/expressive.md",
    theory="docs/theory/22_expressive_gp.md",
    notebook="examples/notebooks/24_expressive_surrogate_gp.ipynb",
    strategy="expressive GP surrogate",
)
_register_family(
    ("NGBoostSurrogate",),
    guide="docs/models/non_gp.md",
    theory="docs/theory/23_non_gp_surrogates.md",
    notebook="examples/notebooks/25_non_gp_surrogates.ipynb",
    strategy="optional NGBoost Gaussian predictive-distribution adapter",
    non_gp=True,
)

_register_family(
    (
        "RandomForestSurrogate",
        "ExtraTreesSurrogate",
        "GradientBoostingSurrogate",
        "HistGradientBoostingSurrogate",
    ),
    guide="docs/models/non_gp.md",
    theory="docs/theory/23_non_gp_surrogates.md",
    notebook="examples/notebooks/25_non_gp_surrogates.ipynb",
    strategy="non-GP tree ensemble surrogate",
    non_gp=True,
)


def get_model_registry_entry(model_name: str) -> ModelRegistryEntry:
    """Return registry metadata for a registered public model."""
    return MODEL_REGISTRY[model_name]
