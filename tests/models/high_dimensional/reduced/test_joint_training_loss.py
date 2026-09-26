import torch

from robotorchan.models import HybridAutoEncoderGP, JointEncoderGP, JointVAEGP


def _data():
    torch.manual_seed(1201)
    X = torch.rand(12, 6, dtype=torch.double)
    Y = X[:, :1] - X[:, 1:2] + 0.5 * X[:, 2:3]
    return X, Y


def test_joint_models_expose_scalar_training_loss():
    X, Y = _data()
    models = [
        JointEncoderGP(X, Y, latent_dim=2, hidden_dims=(6,)),
        HybridAutoEncoderGP(X, Y, latent_dim=2, hidden_dims=(6,)),
        JointVAEGP(X, Y, latent_dim=2, hidden_dims=(6,)),
    ]

    for model in models:
        loss = model.training_loss()
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_joint_training_loss_supports_optimizer_step():
    X, Y = _data()
    model = HybridAutoEncoderGP(X, Y, latent_dim=2, hidden_dims=(6,))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = next(model.encoder.parameters()).detach().clone()
    optimizer.zero_grad()
    model.training_loss().backward()
    optimizer.step()
    after = next(model.encoder.parameters()).detach()

    assert not torch.equal(before, after)
