"""Public registry facade for classification surrogate models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

from torch import nn

from robotorchan.models.capabilities import (
    HighDimensionalStrategy,
    InferenceType,
    InputType,
    ModelCapabilities,
    ObservationType,
    PosteriorSamplingType,
    RobustnessType,
    TaskType,
)
from robotorchan.models.classification.binary.registry import (
    BINARY_CLASSIFICATION_MODEL_SPECS,
)
from robotorchan.models.classification.expressive.joint_encoder import (
    JointEncoderMulticlassGPClassifier,
    JointEncoderOrdinalGPClassifier,
)
from robotorchan.models.classification.expressive.nonstationary import (
    MixedNonstationaryMulticlassGPClassifier,
    MixedNonstationaryOrdinalGPClassifier,
    NonstationaryMulticlassGPClassifier,
    NonstationaryOrdinalGPClassifier,
)
from robotorchan.models.classification.expressive.spectral_mixture import (
    MixedSpectralMixtureMulticlassGPClassifier,
    MixedSpectralMixtureOrdinalGPClassifier,
    SpectralMixtureMulticlassGPClassifier,
    SpectralMixtureOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.deep_gp import (
    MulticlassSingleTaskDeepGPClassifier,
)
from robotorchan.models.classification.multiclass.ensemble import (
    BootstrapGradientBoostingMulticlassClassifier,
    BootstrapGradientBoostingOrdinalClassifier,
    MulticlassClassificationEnsemble,
    OrdinalClassificationEnsemble,
)
from robotorchan.models.classification.multiclass.high_dimensional.alebo import (
    ALEBOMulticlassGPClassifier,
    ALEBOOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.high_dimensional.reduced import (
    PCAMulticlassGPClassifier,
    PCAOrdinalGPClassifier,
    PLSMulticlassGPClassifier,
    PLSOrdinalGPClassifier,
    RandomProjectionMulticlassGPClassifier,
    RandomProjectionOrdinalGPClassifier,
    ReducedMulticlassGPClassifier,
    ReducedOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.high_dimensional.saas import (
    MapSaasMulticlassGPClassifier,
    MapSaasOrdinalGPClassifier,
    SaasMulticlassGPClassifier,
    SaasOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.non_gp import (
    ExtraTreesMulticlassClassifier,
    ExtraTreesOrdinalClassifier,
    GradientBoostingMulticlassClassifier,
    GradientBoostingOrdinalClassifier,
    HistGradientBoostingMulticlassClassifier,
    HistGradientBoostingOrdinalClassifier,
    RandomForestMulticlassClassifier,
    RandomForestOrdinalClassifier,
)
from robotorchan.models.classification.multiclass.robust.label_noise import (
    ContaminatedMulticlassGPClassifier,
    ContaminatedOrdinalGPClassifier,
    LabelNoiseMulticlassGPClassifier,
    LabelNoiseOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.standard.contextual import (
    ContextualMulticlassGPClassifier,
    ContextualOrdinalGPClassifier,
    HierarchicalMulticlassGPClassifier,
    HierarchicalOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.standard.infinite_width_bnn import (
    InfiniteWidthBNNMulticlassGPClassifier,
    InfiniteWidthBNNOrdinalGPClassifier,
    MixedInfiniteWidthBNNMulticlassGPClassifier,
    MixedInfiniteWidthBNNOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.standard.multi_fidelity import (
    MixedMultiFidelityMulticlassGPClassifier,
    MixedMultiFidelityOrdinalGPClassifier,
    MultiFidelityMulticlassGPClassifier,
    MultiFidelityOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.standard.multitask import (
    KroneckerMultiTaskMulticlassGPClassifier,
    KroneckerMultiTaskOrdinalGPClassifier,
    MultiTaskMulticlassGPClassifier,
    MultiTaskOrdinalGPClassifier,
)
from robotorchan.models.classification.multiclass.standard.single_task import (
    MixedMulticlassSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
)
from robotorchan.models.classification.multiclass.uncertain_input import (
    ContinuousUncertainInputMulticlassGPClassifier,
    ContinuousUncertainInputOrdinalGPClassifier,
    UncertainCategoricalMulticlassGPClassifier,
    UncertainCategoricalOrdinalGPClassifier,
)
from robotorchan.models.classification.ordinal.deep_gp import (
    OrdinalSingleTaskDeepGPClassifier,
)
from robotorchan.models.classification.ordinal.standard.single_task import (
    MixedOrdinalSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)

ClassificationModelType: TypeAlias = type[nn.Module]


@dataclass(frozen=True, slots=True)
class ClassificationModelRegistryEntry:
    """Stable public identifier and constructor for a classification model."""

    model_id: str
    model_class: ClassificationModelType
    num_classes: int | None
    family: str
    capabilities: ModelCapabilities


ClassificationModelSpec: TypeAlias = tuple[
    str,
    ClassificationModelType,
    str,
    ModelCapabilities,
]


def _register_fragment(
    registry: dict[str, ClassificationModelRegistryEntry],
    specs: tuple[ClassificationModelSpec, ...],
    *,
    num_classes: int | None,
) -> None:
    """Register one task-specific model fragment without task assumptions."""
    for model_id, model_class, family, capabilities in specs:
        if model_id in registry:
            raise RuntimeError(f"Duplicate classification model ID: {model_id!r}")
        registry[model_id] = ClassificationModelRegistryEntry(
            model_id=model_id,
            model_class=model_class,
            num_classes=num_classes,
            family=family,
            capabilities=capabilities,
        )


def _make_registry() -> dict[str, ClassificationModelRegistryEntry]:
    """Compose task-specific classification registry fragments."""
    registry: dict[str, ClassificationModelRegistryEntry] = {}
    _register_fragment(
        registry,
        BINARY_CLASSIFICATION_MODEL_SPECS,
        num_classes=2,
    )
    _register_fragment(
        registry,
        (
            (
                "multiclass.standard",
                MulticlassSingleTaskGPClassifier,
                "standard",
                ModelCapabilities(
                    observation_type=ObservationType.CLASSIFICATION,
                    inference=InferenceType.VARIATIONAL,
                    supports_multi_output=True,
                    supports_posterior_samples=True,
                    supports_probability_samples=True,
                    posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
                ),
            ),
            (
                "multiclass.mixed",
                MixedMulticlassSingleTaskGPClassifier,
                "mixed",
                ModelCapabilities(
                    observation_type=ObservationType.CLASSIFICATION,
                    input_type=InputType.MIXED,
                    inference=InferenceType.VARIATIONAL,
                    supports_multi_output=True,
                    supports_posterior_samples=True,
                    supports_probability_samples=True,
                    posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
                ),
            ),
        ),
        num_classes=None,
    )
    _register_fragment(
        registry,
        (
            (
                "ordinal.mixed",
                MixedOrdinalSingleTaskGPClassifier,
                "mixed",
                ModelCapabilities(
                    observation_type=ObservationType.CLASSIFICATION,
                    input_type=InputType.MIXED,
                    inference=InferenceType.VARIATIONAL,
                    supports_posterior_samples=True,
                    supports_probability_samples=True,
                    posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
                ),
            ),
            (
                "ordinal.standard",
                OrdinalSingleTaskGPClassifier,
                "ordinal",
                ModelCapabilities(
                    observation_type=ObservationType.CLASSIFICATION,
                    inference=InferenceType.VARIATIONAL,
                    supports_posterior_samples=True,
                    supports_probability_samples=True,
                    posterior_sampling_type=PosteriorSamplingType.GAUSSIAN,
                ),
            ),
        ),
        num_classes=None,
    )
    _register_extended_models(registry)
    return registry


def _register_extended_models(registry: dict[str, ClassificationModelRegistryEntry]) -> None:
    """Register explicit paired implementations using their standard GP contracts."""
    pairs = (
        (
            "alebo",
            ALEBOMulticlassGPClassifier,
            ALEBOOrdinalGPClassifier,
            "high_dimensional",
            {"high_dimensional": HighDimensionalStrategy.RANDOM_EMBEDDING},
        ),
        (
            "joint_encoder",
            JointEncoderMulticlassGPClassifier,
            JointEncoderOrdinalGPClassifier,
            "expressive",
            {"high_dimensional": HighDimensionalStrategy.NEURAL_REDUCTION},
        ),
        (
            "contextual",
            ContextualMulticlassGPClassifier,
            ContextualOrdinalGPClassifier,
            "standard",
            {},
        ),
        (
            "hierarchical",
            HierarchicalMulticlassGPClassifier,
            HierarchicalOrdinalGPClassifier,
            "standard",
            {},
        ),
        (
            "deep_gp",
            MulticlassSingleTaskDeepGPClassifier,
            OrdinalSingleTaskDeepGPClassifier,
            "expressive",
            {
                "high_dimensional": HighDimensionalStrategy.DEEP,
                "posterior_sampling_type": PosteriorSamplingType.STOCHASTIC,
            },
        ),
        (
            "ensemble",
            MulticlassClassificationEnsemble,
            OrdinalClassificationEnsemble,
            "ensemble",
            {
                "inference": InferenceType.NOT_APPLICABLE,
                "ensemble_posterior": True,
                "posterior_sampling_type": PosteriorSamplingType.ENSEMBLE,
            },
        ),
        (
            "bootstrap_gradient_boosting",
            BootstrapGradientBoostingMulticlassClassifier,
            BootstrapGradientBoostingOrdinalClassifier,
            "ensemble",
            {
                "inference": InferenceType.NOT_APPLICABLE,
                "non_gp": True,
                "ensemble_posterior": True,
                "posterior_sampling_type": PosteriorSamplingType.ENSEMBLE,
            },
        ),
        (
            "pca",
            PCAMulticlassGPClassifier,
            PCAOrdinalGPClassifier,
            "reduction",
            {"high_dimensional": HighDimensionalStrategy.REDUCTION},
        ),
        (
            "pls",
            PLSMulticlassGPClassifier,
            PLSOrdinalGPClassifier,
            "reduction",
            {"high_dimensional": HighDimensionalStrategy.REDUCTION},
        ),
        (
            "random_projection",
            RandomProjectionMulticlassGPClassifier,
            RandomProjectionOrdinalGPClassifier,
            "reduction",
            {"high_dimensional": HighDimensionalStrategy.REDUCTION},
        ),
        (
            "reduced",
            ReducedMulticlassGPClassifier,
            ReducedOrdinalGPClassifier,
            "reduction",
            {"high_dimensional": HighDimensionalStrategy.REDUCTION},
        ),
        (
            "saas",
            SaasMulticlassGPClassifier,
            SaasOrdinalGPClassifier,
            "saas",
            {"high_dimensional": HighDimensionalStrategy.SAAS},
        ),
        (
            "map_saas",
            MapSaasMulticlassGPClassifier,
            MapSaasOrdinalGPClassifier,
            "map_saas",
            {"high_dimensional": HighDimensionalStrategy.MAP_SAAS},
        ),
        (
            "multi_fidelity",
            MultiFidelityMulticlassGPClassifier,
            MultiFidelityOrdinalGPClassifier,
            "multi_fidelity",
            {"multi_fidelity": True},
        ),
        (
            "mixed_multi_fidelity",
            MixedMultiFidelityMulticlassGPClassifier,
            MixedMultiFidelityOrdinalGPClassifier,
            "mixed_multi_fidelity",
            {"multi_fidelity": True, "input_type": InputType.MIXED},
        ),
        (
            "multi_task",
            MultiTaskMulticlassGPClassifier,
            MultiTaskOrdinalGPClassifier,
            "multitask",
            {"task_type": TaskType.MULTITASK},
        ),
        (
            "kronecker_multi_task",
            KroneckerMultiTaskMulticlassGPClassifier,
            KroneckerMultiTaskOrdinalGPClassifier,
            "block_multitask",
            {"task_type": TaskType.MULTITASK, "structured_output": True},
        ),
        (
            "random_forest",
            RandomForestMulticlassClassifier,
            RandomForestOrdinalClassifier,
            "non_gp",
            {"inference": InferenceType.NOT_APPLICABLE, "non_gp": True},
        ),
        (
            "extra_trees",
            ExtraTreesMulticlassClassifier,
            ExtraTreesOrdinalClassifier,
            "non_gp",
            {"inference": InferenceType.NOT_APPLICABLE, "non_gp": True},
        ),
        (
            "gradient_boosting",
            GradientBoostingMulticlassClassifier,
            GradientBoostingOrdinalClassifier,
            "non_gp",
            {"inference": InferenceType.NOT_APPLICABLE, "non_gp": True},
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingMulticlassClassifier,
            HistGradientBoostingOrdinalClassifier,
            "non_gp",
            {"inference": InferenceType.NOT_APPLICABLE, "non_gp": True},
        ),
        (
            "nonstationary",
            NonstationaryMulticlassGPClassifier,
            NonstationaryOrdinalGPClassifier,
            "nonstationary",
            {"robustness": frozenset({RobustnessType.NONSTATIONARY})},
        ),
        (
            "mixed_nonstationary",
            MixedNonstationaryMulticlassGPClassifier,
            MixedNonstationaryOrdinalGPClassifier,
            "mixed_nonstationary",
            {
                "input_type": InputType.MIXED,
                "robustness": frozenset({RobustnessType.NONSTATIONARY}),
            },
        ),
        (
            "spectral_mixture",
            SpectralMixtureMulticlassGPClassifier,
            SpectralMixtureOrdinalGPClassifier,
            "spectral_mixture",
            {},
        ),
        (
            "mixed_spectral_mixture",
            MixedSpectralMixtureMulticlassGPClassifier,
            MixedSpectralMixtureOrdinalGPClassifier,
            "mixed_spectral_mixture",
            {"input_type": InputType.MIXED},
        ),
        (
            "infinite_width_bnn",
            InfiniteWidthBNNMulticlassGPClassifier,
            InfiniteWidthBNNOrdinalGPClassifier,
            "infinite_width_bnn",
            {},
        ),
        (
            "mixed_infinite_width_bnn",
            MixedInfiniteWidthBNNMulticlassGPClassifier,
            MixedInfiniteWidthBNNOrdinalGPClassifier,
            "mixed_infinite_width_bnn",
            {"input_type": InputType.MIXED},
        ),
        (
            "label_noise",
            LabelNoiseMulticlassGPClassifier,
            LabelNoiseOrdinalGPClassifier,
            "label_noise",
            {"robustness": frozenset({RobustnessType.LABEL_NOISE})},
        ),
        (
            "contaminated",
            ContaminatedMulticlassGPClassifier,
            ContaminatedOrdinalGPClassifier,
            "contaminated",
            {"robustness": frozenset({RobustnessType.CONTAMINATION})},
        ),
        (
            "continuous_uncertain_input",
            ContinuousUncertainInputMulticlassGPClassifier,
            ContinuousUncertainInputOrdinalGPClassifier,
            "uncertain_input",
            {"robustness": frozenset({RobustnessType.UNCERTAIN_INPUT})},
        ),
        (
            "uncertain_categorical",
            UncertainCategoricalMulticlassGPClassifier,
            UncertainCategoricalOrdinalGPClassifier,
            "uncertain_categorical",
            {
                "input_type": InputType.MIXED,
                "robustness": frozenset({RobustnessType.UNCERTAIN_INPUT}),
            },
        ),
    )
    for suffix, multiclass, ordinal, family, overrides in pairs:
        for kind, model_class in (("multiclass", multiclass), ("ordinal", ordinal)):
            base = registry[f"{kind}.standard"].capabilities
            if family == "non_gp":
                base = replace(
                    base,
                    supports_posterior_samples=False,
                    supports_probability_samples=False,
                    posterior_sampling_type=PosteriorSamplingType.NONE,
                )
            if family == "block_multitask":
                base = replace(base, supports_multi_output=True)
            _register_fragment(
                registry,
                ((f"{kind}.{suffix}", model_class, family, replace(base, **overrides)),),
                num_classes=None,
            )


CLASSIFICATION_MODEL_REGISTRY = _make_registry()


def get_classification_model_entry(model_id: str) -> ClassificationModelRegistryEntry:
    """Return metadata and constructor for a stable classification model ID."""
    try:
        return CLASSIFICATION_MODEL_REGISTRY[model_id]
    except KeyError as error:
        available = ", ".join(CLASSIFICATION_MODEL_REGISTRY)
        message = f"Unknown classification model {model_id!r}. Available: {available}"
        raise KeyError(message) from error


def get_classification_model_class(model_id: str) -> ClassificationModelType:
    """Return the constructor class associated with a stable model ID."""
    return get_classification_model_entry(model_id).model_class


def list_classification_models(*, prefix: str | None = None) -> tuple[str, ...]:
    """List stable public classification model IDs, optionally filtered by prefix."""
    names = tuple(CLASSIFICATION_MODEL_REGISTRY)
    if prefix is None:
        return names
    return tuple(name for name in names if name.startswith(prefix))
