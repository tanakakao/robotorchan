import pytest
import torch

from robotorchan.models.supervised_neural_reduction import SupervisedAutoEncoderInputReducer


def _data():
    torch.manual_seed(601)
    X = torch.rand(24, 8, dtype=torch.double)
    Y = torch.cat(
        [
            (2.0 * X[:, :1] - X[:, 1:2]),
            torch.sin(torch.pi * X[:, 2:3]),
        ],
        dim=-1,
    )
    return X, Y


def _make_reducer(**kwargs):
    return SupervisedAutoEncoderInputReducer(
        latent_dim=3,
        hidden_dims=(10,),
        epochs=12,
        learning_rate=5e-3,
        random_state=23,
        **kwargs,
    )


def test_supervised_autoencoder_requires_y():
    X, _ = _data()
    with pytest.raises(ValueError, match="requires paired Y"):
        _make_reducer().fit(X)


def test_supervised_autoencoder_fits_and_preserves_batch_shape():
    X, Y = _data()
    reducer = _make_reducer().fit(X, Y)

    assert reducer.is_fitted
    assert reducer.input_dim == 8
    assert reducer.output_dim == 3
    assert reducer.transform(X).shape == (24, 3)
    assert reducer.transform(X[:12].reshape(2, 3, 2, 8)).shape == (2, 3, 2, 3)
    assert torch.isfinite(reducer.reconstruction_loss)
    assert torch.isfinite(reducer.supervised_loss)
    assert torch.isfinite(reducer.combined_loss)


def test_supervised_autoencoder_freezes_all_pretraining_modules():
    X, Y = _data()
    reducer = _make_reducer().fit(X, Y)

    assert reducer.encoder is not None
    assert reducer.decoder is not None
    assert reducer.supervised_head is not None
    modules = (reducer.encoder, reducer.decoder, reducer.supervised_head)
    assert all(not parameter.requires_grad for module in modules for parameter in module.parameters())


def test_supervised_autoencoder_transform_keeps_input_gradients():
    X, Y = _data()
    reducer = _make_reducer().fit(X, Y)
    candidate = X[:4].clone().requires_grad_(True)

    reducer.transform(candidate).square().sum().backward()

    assert candidate.grad is not None
    assert candidate.grad.shape == candidate.shape
    assert torch.isfinite(candidate.grad).all()


def test_supervised_autoencoder_auxiliary_prediction_shape():
    X, Y = _data()
    reducer = _make_reducer().fit(X, Y)

    prediction = reducer.predict_auxiliary(X[:5])
    assert prediction.shape == (5, 2)
    assert torch.isfinite(prediction).all()


def test_supervised_autoencoder_state_dict_round_trip():
    X, Y = _data()
    source = _make_reducer().fit(X, Y)
    target = _make_reducer()

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(target.transform(X), source.transform(X))
    torch.testing.assert_close(target.predict_auxiliary(X), source.predict_auxiliary(X))
    assert target.is_fitted


def test_supervised_weight_validation():
    with pytest.raises(ValueError, match="supervised_weight"):
        _make_reducer(supervised_weight=-1.0)
