"""Ownership contracts for task-specific classification registries."""

from robotorchan.models.classification import (
    CLASSIFICATION_MODEL_REGISTRY,
    BinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.registry import (
    BINARY_CLASSIFICATION_MODEL_SPECS,
)


def test_binary_registry_specs_are_owned_by_binary_package() -> None:
    model_ids = tuple(spec[0] for spec in BINARY_CLASSIFICATION_MODEL_SPECS)

    assert model_ids
    assert all(model_id.startswith("binary.") for model_id in model_ids)


def test_shared_registry_composes_binary_specs() -> None:
    model_ids = tuple(spec[0] for spec in BINARY_CLASSIFICATION_MODEL_SPECS)

    assert tuple(CLASSIFICATION_MODEL_REGISTRY) == (
        *model_ids,
        "multiclass.standard",
        "multiclass.mixed",
        "ordinal.mixed",
        "ordinal.standard",
        "multiclass.alebo",
        "ordinal.alebo",
        "multiclass.joint_encoder",
        "ordinal.joint_encoder",
        "multiclass.contextual",
        "ordinal.contextual",
        "multiclass.hierarchical",
        "ordinal.hierarchical",
        "multiclass.deep_gp",
        "ordinal.deep_gp",
        "multiclass.ensemble",
        "ordinal.ensemble",
        "multiclass.bootstrap_gradient_boosting",
        "ordinal.bootstrap_gradient_boosting",
        "multiclass.pca",
        "ordinal.pca",
        "multiclass.pls",
        "ordinal.pls",
        "multiclass.random_projection",
        "ordinal.random_projection",
        "multiclass.reduced",
        "ordinal.reduced",
        "multiclass.saas",
        "ordinal.saas",
        "multiclass.map_saas",
        "ordinal.map_saas",
        "multiclass.multi_fidelity",
        "ordinal.multi_fidelity",
        "multiclass.mixed_multi_fidelity",
        "ordinal.mixed_multi_fidelity",
        "multiclass.multi_task",
        "ordinal.multi_task",
        "multiclass.kronecker_multi_task",
        "ordinal.kronecker_multi_task",
        "multiclass.random_forest",
        "ordinal.random_forest",
        "multiclass.extra_trees",
        "ordinal.extra_trees",
        "multiclass.gradient_boosting",
        "ordinal.gradient_boosting",
        "multiclass.hist_gradient_boosting",
        "ordinal.hist_gradient_boosting",
        "multiclass.nonstationary",
        "ordinal.nonstationary",
        "multiclass.mixed_nonstationary",
        "ordinal.mixed_nonstationary",
        "multiclass.spectral_mixture",
        "ordinal.spectral_mixture",
        "multiclass.mixed_spectral_mixture",
        "ordinal.mixed_spectral_mixture",
        "multiclass.infinite_width_bnn",
        "ordinal.infinite_width_bnn",
        "multiclass.mixed_infinite_width_bnn",
        "ordinal.mixed_infinite_width_bnn",
        "multiclass.label_noise",
        "ordinal.label_noise",
        "multiclass.contaminated",
        "ordinal.contaminated",
        "multiclass.continuous_uncertain_input",
        "ordinal.continuous_uncertain_input",
        "multiclass.uncertain_categorical",
        "ordinal.uncertain_categorical",
    )
    assert CLASSIFICATION_MODEL_REGISTRY["binary.standard"].model_class is (
        BinarySingleTaskGPClassifier
    )


def test_binary_specs_do_not_construct_public_registry_entries() -> None:
    assert all(len(spec) == 4 for spec in BINARY_CLASSIFICATION_MODEL_SPECS)
