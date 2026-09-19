"""Tests for mixed Gaussian uncertain training inputs."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models import MixedUncertainInputSingleTaskGP


def _data():
    X = torch.tensor(
        [[0.0, 0.0, 1.0], [0.3, 1.0, 0.7], [0.7, 0.0, 0.3], [1.0, 1.0, 0.0]],
        dtype=torch.double,
    )
    Y = torch.sin(torch.pi * X[:, :1])
    return X, Y


def test_mixed_uncertain_input_diagonal_contract_and_qucb() -> None:
    X, Y = _data()
    std = torch.full((4, 2), 0.05, dtype=torch.double)
    model = MixedUncertainInputSingleTaskGP(X, Y, cat_dims=[1], train_X_std=std)
    assert model.cat_dims == [1]
    assert torch.equal(model.raw_train_X, X)
    assert torch.equal(model.raw_train_X_std, std)
    assert model.supports_mll
    assert model.make_mll() is not None
    posterior = model.posterior(X[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    acq = qUpperConfidenceBound(model=model, beta=0.2)
    assert torch.isfinite(acq(X[:2].unsqueeze(0))).all()


def test_mixed_uncertain_input_full_covariance_matches_diagonal() -> None:
    X, Y = _data()
    std = torch.full((4, 2), 0.05, dtype=torch.double)
    diagonal = MixedUncertainInputSingleTaskGP(X, Y, cat_dims=[1], train_X_std=std)
    full = MixedUncertainInputSingleTaskGP(
        X, Y, cat_dims=[1], train_X_covar=torch.diag_embed(std.square())
    )
    full.covar_module.load_state_dict(diagonal.covar_module.state_dict())
    left = diagonal.covar_module(diagonal.train_inputs[0], diagonal.train_inputs[0]).to_dense()
    right = full.covar_module(full.train_inputs[0], full.train_inputs[0]).to_dense()
    assert torch.allclose(left, right)


def test_mixed_uncertain_input_supports_multiple_categories() -> None:
    X, Y = _data()
    model = MixedUncertainInputSingleTaskGP(
        X,
        Y,
        cat_dims=[0, 1],
        train_X_std=torch.full((4, 1), 0.05, dtype=torch.double),
    )
    assert model.cat_dims == [0, 1]
    assert torch.isfinite(model.posterior(X[:2]).mean).all()


def test_mixed_uncertain_input_rejects_uncertainty_over_category_width() -> None:
    X, Y = _data()
    try:
        MixedUncertainInputSingleTaskGP(
            X, Y, cat_dims=[1], train_X_std=torch.full_like(X, 0.05)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Uncertainty width must cover continuous dimensions only.")


def test_mixed_uncertain_input_rejects_all_categorical_inputs() -> None:
    X, Y = _data()
    try:
        MixedUncertainInputSingleTaskGP(
            X,
            Y,
            cat_dims=[0, 1, 2],
            train_X_std=torch.empty(4, 0, dtype=torch.double),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("At least one continuous feature is required.")
