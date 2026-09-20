"""Tests for infinite-width neural-network kernel GP models."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models.infinite_width_bnn import InfiniteWidthBNNGP, InfiniteWidthReLUKernel


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    X = torch.rand(12, 3, dtype=torch.double)
    Y = torch.sin(5.0 * X[:, :1]) + X[:, 1:2] * X[:, 2:3]
    return X, Y


def test_infinite_width_relu_kernel_is_symmetric_and_finite() -> None:
    X, _ = _data()
    kernel = InfiniteWidthReLUKernel(depth=3, ard_num_dims=3).to(X)

    covariance = kernel(X, X).to_dense()

    assert covariance.shape == (12, 12)
    assert torch.allclose(covariance, covariance.transpose(-1, -2), atol=1e-10)
    assert torch.isfinite(covariance).all()
    assert torch.all(covariance.diagonal() > 0)


def test_infinite_width_bnn_gp_uses_common_exact_gp_contract() -> None:
    X, Y = _data()
    model = InfiniteWidthBNNGP(X, Y, depth=2)

    assert torch.equal(model.raw_train_X, X)
    assert torch.equal(model.raw_train_Y, Y)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    assert model.depth == 2


def test_infinite_width_bnn_gp_posterior_and_qlogei_are_finite() -> None:
    X, Y = _data()
    model = InfiniteWidthBNNGP(X, Y, depth=2)
    model.eval()

    posterior = model.posterior(X[:3])
    acquisition = qLogExpectedImprovement(model=model, best_f=Y.max())
    value = acquisition(X[:2].unsqueeze(0))

    assert posterior.mean.shape == (3, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(value).all()


def test_infinite_width_bnn_gp_optimize_acqf_runs() -> None:
    X, Y = _data()
    model = InfiniteWidthBNNGP(X, Y, depth=2)
    model.eval()
    acquisition = qLogExpectedImprovement(model=model, best_f=Y.max())
    bounds = torch.stack((torch.zeros(3, dtype=X.dtype), torch.ones(3, dtype=X.dtype)))

    candidate, value = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=12,
        options={"maxiter": 12},
    )

    assert candidate.shape == (1, 3)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_infinite_width_bnn_kernel_validates_hyperparameters() -> None:
    for kwargs in (
        {"depth": 0},
        {"weight_variance": 0.0},
        {"bias_variance": -0.1},
        {"eps": 0.0},
    ):
        try:
            InfiniteWidthReLUKernel(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {kwargs}.")
