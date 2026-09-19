"""Tests for uncertain training-input GP."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models.uncertain_input import UncertainInputSingleTaskGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0, 1, 10, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(2 * torch.pi * X)
    return X, Y


def test_uncertain_input_model_retains_uncertainty() -> None:
    X, Y = _data()
    std = torch.full_like(X, 0.05)
    model = UncertainInputSingleTaskGP(X, Y, train_X_std=std)
    assert torch.equal(model.raw_train_X, X)
    assert torch.equal(model.raw_train_X_std, std)


def test_zero_uncertainty_kernel_matches_rbf_formula() -> None:
    X, Y = _data()
    model = UncertainInputSingleTaskGP(X, Y, train_X_std=torch.zeros_like(X))
    kernel = model.covar_module
    augmented = torch.cat([X, torch.zeros_like(X)], dim=-1)
    actual = kernel(augmented, augmented).to_dense()
    lengthscale = kernel.lengthscale
    delta2 = (X.unsqueeze(-2) - X.unsqueeze(-3)).square()
    expected = torch.exp(-0.5 * (delta2 / lengthscale.square()).sum(dim=-1))
    assert torch.allclose(actual, expected)


def test_input_uncertainty_changes_training_covariance() -> None:
    X, Y = _data()
    zero = UncertainInputSingleTaskGP(X, Y, train_X_std=torch.zeros_like(X))
    noisy = UncertainInputSingleTaskGP(X, Y, train_X_std=torch.full_like(X, 0.2))
    noisy.covar_module.load_state_dict(zero.covar_module.state_dict())
    zero_cov = zero.covar_module(zero.train_inputs[0], zero.train_inputs[0]).to_dense()
    noisy_cov = noisy.covar_module(noisy.train_inputs[0], noisy.train_inputs[0]).to_dense()
    assert not torch.allclose(zero_cov, noisy_cov)


def test_uncertain_input_posterior_and_acquisition() -> None:
    X, Y = _data()
    model = UncertainInputSingleTaskGP(X, Y, train_X_std=torch.full_like(X, 0.05))
    posterior = model.posterior(X[:3])
    assert posterior.mean.shape == torch.Size([3, 1])
    acquisition = qUpperConfidenceBound(model=model, beta=0.2)
    value = acquisition(X[:3].unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_uncertain_input_validation() -> None:
    X, Y = _data()
    try:
        UncertainInputSingleTaskGP(X, Y, train_X_std=-torch.ones_like(X))
    except ValueError:
        pass
    else:
        raise AssertionError("Negative input uncertainty must be rejected.")
