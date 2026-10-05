"""Cross-family tests for robust binary classification."""

import torch

from robotorchan.models.capabilities import InputType, RobustnessType, TaskType
from robotorchan.models.classification import (
    CLASSIFICATION_MODEL_REGISTRY,
    ClassificationModelList,
    ContaminatedBinarySingleTaskGPClassifier,
    LabelNoiseBinarySingleTaskGPClassifier,
    MixedNonstationaryBinarySingleTaskGPClassifier,
    NonstationaryBinarySingleTaskGPClassifier,
    NonstationaryMultiTaskBinaryGPClassifier,
)


def _binary_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.tensor(
        [[0.0, 0.0], [0.2, 1.0], [0.6, 0.0], [0.9, 1.0]],
        dtype=torch.double,
    )
    train_y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double)
    return train_x, train_y


def test_mixed_nonstationary_uses_only_continuous_dims_for_local_lengthscale() -> None:
    train_x, train_y = _binary_data()
    model = MixedNonstationaryBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        cat_dims=[1],
    )

    additive, interaction = model.local_lengthscale(train_x)

    assert additive.shape == torch.Size([4, 1])
    assert interaction.shape == torch.Size([4, 1])
    assert model.cat_dims == (1,)
    assert model.continuous_dims == (0,)


def test_multitask_nonstationary_excludes_task_identity_from_local_lengthscale() -> None:
    train_x, train_y = _binary_data()
    model = NonstationaryMultiTaskBinaryGPClassifier(
        train_x,
        train_y,
        task_feature=1,
    )

    lengthscale = model.local_lengthscale(train_x)

    assert lengthscale.shape == torch.Size([4, 1])
    assert model.task_feature == 1
    assert model.continuous_dims == (0,)


def test_model_list_composes_heterogeneous_robust_classifiers_without_wrapper() -> None:
    train_x, train_y = _binary_data()
    label_noise = LabelNoiseBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        flip_probability=0.05,
    )
    contaminated = ContaminatedBinarySingleTaskGPClassifier(
        train_x,
        train_y,
        contamination_rate=0.05,
    )
    model_list = ClassificationModelList(label_noise, contaminated)

    probabilities = model_list.predict_proba(train_x[:2])

    assert len(probabilities) == 2
    assert all(probability.shape == torch.Size([2, 2]) for probability in probabilities)


def test_cross_family_nonstationary_registry_capabilities() -> None:
    mixed = CLASSIFICATION_MODEL_REGISTRY["binary.robust.mixed_nonstationary"]
    multitask = CLASSIFICATION_MODEL_REGISTRY["binary.robust.multitask_nonstationary"]

    assert mixed.capabilities.input_type is InputType.MIXED
    assert RobustnessType.NONSTATIONARY in mixed.capabilities.robustness
    assert multitask.capabilities.task_type is TaskType.MULTITASK
    assert RobustnessType.NONSTATIONARY in multitask.capabilities.robustness


def test_standard_nonstationary_remains_available_for_high_dimensional_composition() -> None:
    train_x, train_y = _binary_data()
    model = NonstationaryBinarySingleTaskGPClassifier(train_x, train_y)

    assert model.local_lengthscale(train_x).shape == train_x.shape
