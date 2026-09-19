import torch

from robotorchan.models import MixedHybridAutoEncoderGP, MixedJointEncoderGP, MixedJointVAEGP


def _data():
    torch.manual_seed(808)
    cont = torch.randn(14, 4, dtype=torch.double)
    cat = torch.randint(0, 3, (14, 1)).to(torch.double)
    X = torch.cat((cont[:, :2], cat, cont[:, 2:]), dim=-1)
    Y = cont[:, :1] - 0.3 * cont[:, 1:2] + 0.2 * cat
    return X, Y


def test_mixed_joint_encoder_keeps_categories_outside_encoder_and_backpropagates():
    X, Y = _data()
    model = MixedJointEncoderGP(X, Y, 2, [2], hidden_dims=(6,), random_state=8)

    assert model.encoder[0].in_features == 4
    assert model.cat_dims == [2]
    assert model.encode(X).shape == (14, 3)
    torch.testing.assert_close(model.encode(X)[..., 2], X[..., 2])
    loss = model.training_loss()
    loss.backward()
    assert model.encoder[0].weight.grad is not None
    assert torch.isfinite(model.encoder[0].weight.grad).all()


def test_mixed_joint_encoder_supports_negative_cat_dims_and_raw_posterior():
    X, Y = _data()
    X = torch.cat((X[:, :2], X[:, 3:], X[:, 2:3]), dim=-1)
    model = MixedJointEncoderGP(X, Y, 2, [-1], hidden_dims=(6,), random_state=8)
    model.eval()
    model.likelihood.eval()

    posterior = model.posterior(X[:3])

    assert model.cat_dims == [4]
    assert posterior.mean.shape == (3, 1)


def test_mixed_hybrid_reconstructs_continuous_columns_only():
    X, Y = _data()
    model = MixedHybridAutoEncoderGP(
        X,
        Y,
        2,
        [2],
        hidden_dims=(6,),
        reconstruction_weight=0.5,
        random_state=8,
    )

    assert model.reconstruct(X[:3]).shape == (3, 4)
    loss = model.training_loss()
    loss.backward()
    assert next(model.decoder.parameters()).grad is not None


def test_mixed_joint_vae_preserves_gp_and_variational_gradients():
    X, Y = _data()
    model = MixedJointVAEGP(
        X,
        Y,
        2,
        [2],
        hidden_dims=(6,),
        beta=0.2,
        reconstruction_weight=0.5,
        random_state=8,
    )

    mu, logvar = model.encode_distribution(X[:3])
    assert mu.shape == (3, 2)
    assert logvar.shape == (3, 2)
    assert model.encode(X[:3]).shape == (3, 3)
    torch.testing.assert_close(model.encode(X[:3])[..., 2], X[:3, 2])
    loss = model.training_loss()
    loss.backward()
    assert model.mu_head.weight.grad is not None
    assert model.logvar_head.weight.grad is not None
    assert next(model.decoder.parameters()).grad is not None
