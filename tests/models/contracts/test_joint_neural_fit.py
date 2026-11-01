"""Native fitting contract for jointly trained neural GP models."""

import pytest
import torch

from robotorchan.models import HybridAutoEncoderGP, JointEncoderGP


@pytest.mark.parametrize("model_type", [JointEncoderGP, HybridAutoEncoderGP])
def test_joint_neural_fit_updates_parameters(model_type: type) -> None:
    torch.manual_seed(0)
    train_X = torch.rand(10, 3)
    train_Y = torch.sin(train_X[:, :1])
    model = model_type(train_X, train_Y, latent_dim=2, hidden_dims=(8,))
    before = [parameter.detach().clone() for parameter in model.parameters()]

    assert model.supports_fit
    assert model.fit(epochs=2, lr=0.01) is None
    assert not model.training
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    assert torch.isfinite(model.posterior(train_X[:2]).mean).all()


@pytest.mark.parametrize("epochs, lr", [(0, 0.01), (1, 0.0)])
def test_joint_neural_fit_rejects_invalid_settings(epochs: int, lr: float) -> None:
    train_X = torch.rand(8, 3)
    model = JointEncoderGP(train_X, train_X[:, :1], latent_dim=2, hidden_dims=(8,))
    with pytest.raises(ValueError):
        model.fit(epochs=epochs, lr=lr)
