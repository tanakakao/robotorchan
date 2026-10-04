"""State-dict serialization contracts for classification models."""

import torch

from robotorchan.models.classification import (
    BinarySingleTaskGPClassifier,
    ClassificationModelList,
    JointEncoderBinaryGPClassifier,
    MixedBinarySingleTaskGPClassifier,
    MultiTaskBinaryGPClassifier,
    PCABinarySingleTaskGPClassifier,
)


def _binary_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [[0.05, 0.10], [0.20, 0.30], [0.35, 0.20], [0.65, 0.80], [0.80, 0.70], [0.95, 0.90]],
        dtype=torch.double,
    )
    train_Y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.double)
    test_X = torch.tensor([[0.25, 0.25], [0.75, 0.75]], dtype=torch.double)
    return train_X, train_Y, test_X


def _assert_prediction_round_trip(source: object, restored: object, X: torch.Tensor) -> None:
    source.eval()
    restored.eval()
    with torch.no_grad():
        expected_latent = source.latent_posterior(X)
        actual_latent = restored.latent_posterior(X)
        expected_probability = source.predict_proba(X)
        actual_probability = restored.predict_proba(X)
    torch.testing.assert_close(actual_latent.mean, expected_latent.mean)
    torch.testing.assert_close(actual_latent.variance, expected_latent.variance)
    torch.testing.assert_close(actual_probability, expected_probability)


def test_binary_classifier_state_dict_round_trip() -> None:
    train_X, train_Y, test_X = _binary_data()
    source = BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)
    restored = BinarySingleTaskGPClassifier(train_X, train_Y, inducing_points=4)

    restored.load_state_dict(source.state_dict())

    _assert_prediction_round_trip(source, restored, test_X)


def test_mixed_classifier_state_dict_round_trip() -> None:
    train_X, train_Y, test_X = _binary_data()
    train_X[:, 1] = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.double)
    test_X[:, 1] = torch.tensor([0, 1], dtype=torch.double)
    source = MixedBinarySingleTaskGPClassifier(train_X, train_Y, cat_dims=[1])
    restored = MixedBinarySingleTaskGPClassifier(train_X, train_Y, cat_dims=[1])

    restored.load_state_dict(source.state_dict())

    _assert_prediction_round_trip(source, restored, test_X)
    assert restored.cat_dims == [1]


def test_multitask_classifier_state_dict_round_trip() -> None:
    train_X, train_Y, test_X = _binary_data()
    tasks = torch.tensor([[0], [1], [0], [1], [0], [1]], dtype=torch.double)
    train_X = torch.cat((train_X[:, :1], tasks), dim=-1)
    test_X = torch.tensor([[0.25, 0], [0.75, 1]], dtype=torch.double)
    source = MultiTaskBinaryGPClassifier(train_X, train_Y, task_feature=1)
    restored = MultiTaskBinaryGPClassifier(train_X, train_Y, task_feature=1)

    restored.load_state_dict(source.state_dict())

    _assert_prediction_round_trip(source, restored, test_X)
    assert restored.task_feature == source.task_feature
    assert restored.num_tasks == source.num_tasks


def test_pca_classifier_state_dict_includes_reducer_and_round_trips() -> None:
    train_X, train_Y, test_X = _binary_data()
    source = PCABinarySingleTaskGPClassifier(train_X, train_Y, n_components=1)
    restored = PCABinarySingleTaskGPClassifier(train_X, train_Y, n_components=1)

    state = source.state_dict()
    assert any(key.startswith("input_reducer.") for key in state)
    restored.load_state_dict(state)

    _assert_prediction_round_trip(source, restored, test_X)


def test_joint_encoder_classifier_state_dict_round_trip() -> None:
    train_X, train_Y, test_X = _binary_data()
    source = JointEncoderBinaryGPClassifier(
        train_X,
        train_Y,
        latent_dim=2,
        hidden_dims=(4,),
        random_state=7,
    )
    restored = JointEncoderBinaryGPClassifier(
        train_X,
        train_Y,
        latent_dim=2,
        hidden_dims=(4,),
        random_state=19,
    )

    restored.load_state_dict(source.state_dict())

    _assert_prediction_round_trip(source, restored, test_X)
    torch.testing.assert_close(restored.x_mean, source.x_mean)
    torch.testing.assert_close(restored.x_scale, source.x_scale)


def test_classification_model_list_state_dict_round_trip() -> None:
    train_X, train_Y, test_X = _binary_data()
    source = ClassificationModelList(
        BinarySingleTaskGPClassifier(train_X, train_Y),
        BinarySingleTaskGPClassifier(train_X, 1.0 - train_Y),
    )
    restored = ClassificationModelList(
        BinarySingleTaskGPClassifier(train_X, train_Y),
        BinarySingleTaskGPClassifier(train_X, 1.0 - train_Y),
    )

    restored.load_state_dict(source.state_dict())

    expected = source.predict_proba(test_X)
    actual = restored.predict_proba(test_X)
    for expected_probability, actual_probability in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_probability, expected_probability)
