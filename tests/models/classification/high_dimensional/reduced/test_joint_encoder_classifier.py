"""Tests for joint neural binary GP classification."""

import pytest
import torch

from robotorchan.models.classification import JointEncoderBinaryGPClassifier


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(29)
    X = torch.rand(12, 6, dtype=torch.double)
    Y = (X[:, 0] - X[:, 1] + X[:, 2] > 0.5).to(dtype=torch.double)
    return X, Y


def test_joint_encoder_classifier_preserves_prediction_contract() -> None:
    train_X, train_Y = _data()
    model = JointEncoderBinaryGPClassifier(train_X, train_Y, latent_dim=3)
    probabilities = model.predict_proba(train_X[:4])
    assert probabilities.shape == torch.Size([4, 2])
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4, dtype=torch.double))
    assert model.encode(train_X[:4]).shape == torch.Size([4, 3])


def test_joint_training_loss_reaches_encoder_parameters() -> None:
    train_X, train_Y = _data()
    model = JointEncoderBinaryGPClassifier(train_X, train_Y, latent_dim=3)
    loss = model.training_loss()
    loss.backward()
    gradients = [parameter.grad for parameter in model.encoder.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_joint_encoder_classifier_preserves_raw_training_inputs() -> None:
    train_X, train_Y = _data()
    model = JointEncoderBinaryGPClassifier(train_X, train_Y, latent_dim=3)
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)


def test_joint_encoder_classifier_rejects_wrong_input_dimension() -> None:
    train_X, train_Y = _data()
    model = JointEncoderBinaryGPClassifier(train_X, train_Y, latent_dim=3)
    with pytest.raises(ValueError, match="Expected final dimension"):
        model.predict_proba(torch.rand(2, 5, dtype=torch.double))
