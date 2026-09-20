"""Tests for jointly trained neural multi-task GP models."""

import torch

from robotorchan.models.reduced.joint_multitask import (
    HybridAutoEncoderKroneckerMultiTaskGP,
    HybridAutoEncoderMultiTaskGP,
    JointEncoderKroneckerMultiTaskGP,
    JointEncoderMultiTaskGP,
    JointVAEKroneckerMultiTaskGP,
    JointVAEMultiTaskGP,
)


def _long_data() -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.rand(10, 4, dtype=torch.double)
    task = torch.tensor([0.0, 1.0] * 5, dtype=torch.double).unsqueeze(-1)
    return torch.cat([data[:, :2], task, data[:, 2:]], dim=-1), data[:, :1]


def _block_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.rand(8, 4, dtype=torch.double)
    return X, torch.stack([X[:, 0], X[:, 1]], dim=-1)


def test_joint_encoder_multitask_preserves_task_and_encoder_gradients() -> None:
    train_X, train_Y = _long_data()
    model = JointEncoderMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        latent_dim=2,
        hidden_dims=(6,),
    )
    encoded = model.encode(train_X)
    assert encoded.shape == torch.Size([10, 3])
    assert torch.equal(encoded[:, -1], train_X[:, 2])
    loss = model.training_loss()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_joint_encoder_kronecker_supports_training_loss() -> None:
    train_X, train_Y = _block_data()
    model = JointEncoderKroneckerMultiTaskGP(
        train_X,
        train_Y,
        latent_dim=2,
        hidden_dims=(6,),
    )
    loss = model.training_loss()
    assert loss.ndim == 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_hybrid_multitask_adds_reconstruction_loss() -> None:
    train_X, train_Y = _long_data()
    model = HybridAutoEncoderMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        latent_dim=2,
        hidden_dims=(6,),
        reconstruction_weight=0.5,
    )
    assert model.reconstruction_loss().ndim == 0
    assert model.training_loss().ndim == 0


def test_hybrid_kronecker_adds_reconstruction_loss() -> None:
    train_X, train_Y = _block_data()
    model = HybridAutoEncoderKroneckerMultiTaskGP(
        train_X,
        train_Y,
        latent_dim=2,
        hidden_dims=(6,),
        reconstruction_weight=0.5,
    )
    assert model.reconstruction_loss().ndim == 0


def test_joint_vae_multitask_adds_kl_loss() -> None:
    train_X, train_Y = _long_data()
    model = JointVAEMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        latent_dim=2,
        hidden_dims=(6,),
        beta=0.2,
    )
    assert model.kl_loss().ndim == 0
    assert model.training_loss().ndim == 0


def test_joint_vae_kronecker_adds_kl_loss() -> None:
    train_X, train_Y = _block_data()
    model = JointVAEKroneckerMultiTaskGP(
        train_X,
        train_Y,
        latent_dim=2,
        hidden_dims=(6,),
        beta=0.2,
    )
    assert model.kl_loss().ndim == 0
    assert model.training_loss().ndim == 0


def test_joint_encoder_multitask_supports_custom_expansive_feature_extractor() -> None:
    train_X, train_Y = _long_data()
    extractor = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Tanh())
    model = JointEncoderMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        latent_dim=7,
        feature_extractor=extractor,
    )

    encoded = model.encode(train_X)
    assert encoded.shape == torch.Size([10, 8])
    assert model.encoder is extractor
    torch.testing.assert_close(encoded[:, -1], train_X[:, 2])


def test_joint_encoder_kronecker_supports_custom_expansive_feature_extractor() -> None:
    train_X, train_Y = _block_data()
    extractor = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.GELU())
    model = JointEncoderKroneckerMultiTaskGP(
        train_X,
        train_Y,
        latent_dim=6,
        feature_extractor=extractor,
    )

    assert model.encoder is extractor
    assert model.encode(train_X).shape == torch.Size([8, 6])
    loss = model.training_loss()
    loss.backward()
    assert extractor[0].weight.grad is not None



def test_joint_encoder_multitask_posterior_accepts_original_space() -> None:
    train_X, train_Y = _long_data()
    model = JointEncoderMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        latent_dim=2,
        hidden_dims=(6,),
    )
    model.eval()
    model.likelihood.eval()
    test_X = train_X[:3].detach().clone().requires_grad_(True)
    posterior = model.posterior(test_X)
    assert posterior.mean.shape[-2:] == torch.Size([3, 1])
    posterior.mean.sum().backward()
    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()


def test_joint_encoder_kronecker_posterior_accepts_original_space() -> None:
    train_X, train_Y = _block_data()
    model = JointEncoderKroneckerMultiTaskGP(
        train_X,
        train_Y,
        latent_dim=2,
        hidden_dims=(6,),
    )
    model.eval()
    model.likelihood.eval()
    test_X = train_X[:3].detach().clone().requires_grad_(True)
    posterior = model.posterior(test_X)
    assert posterior.mean.shape[-2:] == torch.Size([3, 2])
    posterior.mean.sum().backward()
    assert test_X.grad is not None
    assert torch.isfinite(test_X.grad).all()
