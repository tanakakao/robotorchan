from __future__ import annotations

import torch

from robotorchan.models.neural_reduction import AutoEncoderInputReducer


def _training_data() -> torch.Tensor:
    torch.manual_seed(203)
    latent = torch.randn(40, 3, dtype=torch.double)
    mixing = torch.randn(3, 8, dtype=torch.double)
    return latent @ mixing + 0.02 * torch.randn(40, 8, dtype=torch.double)


def test_autoencoder_reducer_fit_transform_shape_and_metadata() -> None:
    X = _training_data()
    reducer = AutoEncoderInputReducer(
        latent_dim=3,
        hidden_dims=(12, 6),
        epochs=30,
        learning_rate=5e-3,
        random_state=7,
    )

    Z = reducer.fit_transform(X)

    assert reducer.is_fitted is True
    assert reducer.input_dim == 8
    assert reducer.output_dim == 3
    assert Z.shape == torch.Size([40, 3])
    assert torch.isfinite(Z).all()
    assert torch.isfinite(reducer.reconstruction_loss)


def test_autoencoder_reducer_preserves_leading_dimensions() -> None:
    X = _training_data()
    reducer = AutoEncoderInputReducer(
        latent_dim=2,
        hidden_dims=(10,),
        epochs=20,
        random_state=3,
    ).fit(X)

    candidates = torch.randn(2, 4, 5, 8, dtype=torch.double)
    Z = reducer.transform(candidates)

    assert Z.shape == torch.Size([2, 4, 5, 2])


def test_autoencoder_transform_keeps_input_gradients_and_freezes_network() -> None:
    X = _training_data()
    reducer = AutoEncoderInputReducer(
        latent_dim=3,
        hidden_dims=(10,),
        epochs=20,
        random_state=11,
    ).fit(X)
    assert reducer.encoder is not None
    assert reducer.decoder is not None

    assert all(not parameter.requires_grad for parameter in reducer.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in reducer.decoder.parameters())

    candidates = torch.randn(4, 8, dtype=torch.double, requires_grad=True)
    value = reducer.transform(candidates).square().sum()
    value.backward()

    assert candidates.grad is not None
    assert candidates.grad.shape == candidates.shape
    assert torch.isfinite(candidates.grad).all()


def test_autoencoder_transform_does_not_modify_frozen_parameters() -> None:
    X = _training_data()
    reducer = AutoEncoderInputReducer(
        latent_dim=2,
        hidden_dims=(10,),
        epochs=20,
        random_state=13,
    ).fit(X)
    assert reducer.encoder is not None

    before = [parameter.detach().clone() for parameter in reducer.encoder.parameters()]
    reducer.transform(torch.randn(6, 8, dtype=torch.double))
    after = list(reducer.encoder.parameters())

    for expected, actual in zip(before, after, strict=True):
        torch.testing.assert_close(actual, expected)


def test_autoencoder_reconstructs_original_shape() -> None:
    X = _training_data()
    reducer = AutoEncoderInputReducer(
        latent_dim=3,
        hidden_dims=(12, 6),
        epochs=30,
        learning_rate=5e-3,
        random_state=17,
    ).fit(X)

    reconstructed = reducer.reconstruct(X[:5])

    assert reconstructed.shape == torch.Size([5, 8])
    assert torch.isfinite(reconstructed).all()


def test_autoencoder_reducer_state_dict_round_trip() -> None:
    X = _training_data()
    source = AutoEncoderInputReducer(
        latent_dim=3,
        hidden_dims=(12, 6),
        epochs=20,
        random_state=19,
    ).fit(X)
    target = AutoEncoderInputReducer(
        latent_dim=3,
        hidden_dims=(12, 6),
        epochs=20,
        random_state=19,
    )

    target.load_state_dict(source.state_dict())

    candidates = torch.randn(7, 8, dtype=torch.double)
    torch.testing.assert_close(target.transform(candidates), source.transform(candidates))
    assert target.is_fitted is True
    assert target.encoder is not None
    assert target.decoder is not None
    assert all(not parameter.requires_grad for parameter in target.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in target.decoder.parameters())


def test_autoencoder_reducer_follows_dtype_conversion() -> None:
    X = _training_data()
    reducer = AutoEncoderInputReducer(
        latent_dim=2,
        hidden_dims=(8,),
        epochs=20,
        random_state=23,
    ).fit(X).float()
    assert reducer.encoder is not None
    assert reducer.x_mean is not None
    assert reducer.x_scale is not None

    assert reducer.x_mean.dtype == torch.float32
    assert reducer.x_scale.dtype == torch.float32
    assert next(reducer.encoder.parameters()).dtype == torch.float32

    Z = reducer.transform(torch.randn(3, 8, dtype=torch.float32))
    assert Z.dtype == torch.float32
