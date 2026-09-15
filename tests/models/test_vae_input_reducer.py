from __future__ import annotations

import torch

from robotorchan.models.neural_reduction import VAEInputReducer


def _training_data() -> torch.Tensor:
    torch.manual_seed(401)
    latent = torch.randn(32, 3, dtype=torch.double)
    mixing = torch.randn(3, 8, dtype=torch.double)
    return latent @ mixing + 0.02 * torch.randn(32, 8, dtype=torch.double)


def _make_reducer() -> VAEInputReducer:
    return VAEInputReducer(
        latent_dim=3,
        hidden_dims=(10,),
        epochs=12,
        learning_rate=5e-3,
        beta=0.2,
        random_state=29,
    )


def test_vae_reducer_uses_deterministic_posterior_mean_transform() -> None:
    X = _training_data()
    reducer = _make_reducer().fit(X)

    first = reducer.transform(X[:5])
    second = reducer.transform(X[:5])
    mu, logvar = reducer.encode_distribution(X[:5])

    assert first.shape == torch.Size([5, 3])
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, mu)
    assert logvar.shape == first.shape
    assert torch.isfinite(reducer.reconstruction_loss)
    assert torch.isfinite(reducer.kl_loss)
    assert torch.isfinite(reducer.elbo_loss)


def test_vae_reducer_preserves_gradients_and_supports_latent_sampling() -> None:
    reducer = _make_reducer().fit(_training_data())
    X = torch.randn(4, 8, dtype=torch.double, requires_grad=True)

    reducer.transform(X).square().sum().backward()
    samples = reducer.sample_latent(X.detach(), n_samples=5)

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    assert samples.shape == torch.Size([5, 4, 3])
    assert torch.isfinite(samples).all()


def test_vae_reducer_reconstructs_and_round_trips_state_dict() -> None:
    X = _training_data()
    source = _make_reducer().fit(X)
    target = _make_reducer()
    target.load_state_dict(source.state_dict())

    candidates = X[:6]
    assert source.reconstruct(candidates).shape == candidates.shape
    torch.testing.assert_close(target.transform(candidates), source.transform(candidates))
    source_mu, source_logvar = source.encode_distribution(candidates)
    target_mu, target_logvar = target.encode_distribution(candidates)
    torch.testing.assert_close(target_mu, source_mu)
    torch.testing.assert_close(target_logvar, source_logvar)
