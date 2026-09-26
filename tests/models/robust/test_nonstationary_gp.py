"""Tests for the nonstationary Gibbs-kernel GP."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models.robust.nonstationary import GibbsKernel, NonstationarySingleTaskGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0, 1, 10, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(2 * torch.pi * X)
    return X, Y


def test_local_lengthscales_are_positive() -> None:
    X, _ = _data()
    kernel = GibbsKernel(1, lengthscale_floor=1e-4).double()

    lengthscale = kernel.local_lengthscale(X)

    assert lengthscale.shape == X.shape
    assert torch.all(lengthscale > 0)


def test_kernel_is_symmetric_and_positive_semidefinite() -> None:
    X, _ = _data()
    kernel = GibbsKernel(1).double()
    with torch.no_grad():
        kernel.lengthscale_slope.fill_(1.5)

    covariance = kernel(X, X).to_dense()
    eigenvalues = torch.linalg.eigvalsh(covariance)

    assert torch.allclose(covariance, covariance.transpose(-1, -2))
    assert eigenvalues.min() >= -1e-8


def test_zero_slope_is_stationary_and_slope_changes_covariance() -> None:
    X, _ = _data()
    kernel = GibbsKernel(1).double()
    stationary = kernel(X, X).to_dense()
    with torch.no_grad():
        kernel.lengthscale_slope.fill_(2.0)
    nonstationary = kernel(X, X).to_dense()

    assert not torch.allclose(stationary, nonstationary)


def test_gradients_reach_local_lengthscale_parameters() -> None:
    X, Y = _data()
    model = NonstationarySingleTaskGP(X, Y)
    mll = model.make_mll()
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.gibbs_kernel.lengthscale_intercept.grad is not None
    assert model.gibbs_kernel.lengthscale_slope.grad is not None


def test_posterior_and_qmc_acquisition_contract() -> None:
    X, Y = _data()
    model = NonstationarySingleTaskGP(X, Y)
    candidate = X[:3]

    posterior = model.posterior(candidate)
    assert posterior.mean.shape == torch.Size([3, 1])

    acquisition = qUpperConfidenceBound(model=model, beta=0.2)
    value = acquisition(candidate[:2].unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_raw_training_data_and_lengthscale_diagnostic() -> None:
    X, Y = _data()
    model = NonstationarySingleTaskGP(X, Y)

    assert torch.equal(model.raw_train_X, X)
    assert torch.equal(model.raw_train_Y, Y)
    lengthscale = model.local_lengthscale(X)
    assert lengthscale.shape == X.shape
    assert torch.all(lengthscale > 0)
