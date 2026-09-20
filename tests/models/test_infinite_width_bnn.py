"""Tests for infinite-width neural-network kernel GP models."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import InfiniteWidthBNNGP, InfiniteWidthBNNMultiTaskGP
from robotorchan.models.infinite_width_bnn import InfiniteWidthReLUKernel


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
    eigenvalues = torch.linalg.eigvalsh(covariance)
    assert eigenvalues.min() >= -1e-8


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
    objective = GenericMCObjective(lambda samples, X=None: samples.squeeze(-1))
    acquisition = qLogExpectedImprovement(model=model, best_f=Y.max(), objective=objective)
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


def _multitask_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(37)
    data_X = torch.rand(12, 2, dtype=torch.double)
    tasks = torch.arange(12, dtype=torch.double).remainder(3).unsqueeze(-1)
    train_X = torch.cat((data_X[:, :1], tasks, data_X[:, 1:]), dim=-1)
    train_Y = torch.sin(4.0 * data_X[:, :1]) + 0.3 * data_X[:, 1:] + 0.2 * tasks
    return train_X, train_Y


def test_infinite_width_bnn_multitask_uses_common_contract() -> None:
    X, Y = _multitask_data()
    model = InfiniteWidthBNNMultiTaskGP(X, Y, task_feature=1, depth=2)

    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    assert model.depth == 2
    data_kernel = model.covar_module.kernels[0]
    assert data_kernel.base_kernel.active_dims.tolist() == [0, 2]


def test_infinite_width_bnn_multitask_posterior_is_finite() -> None:
    X, Y = _multitask_data()
    model = InfiniteWidthBNNMultiTaskGP(X, Y, task_feature=1, depth=2)
    model.eval()

    posterior = model.posterior(X[:4])

    assert posterior.mean.shape == (4, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_infinite_width_bnn_multitask_qlogei_runs() -> None:
    X, Y = _multitask_data()
    model = InfiniteWidthBNNMultiTaskGP(X, Y, task_feature=1, depth=2)
    model.eval()
    objective = GenericMCObjective(lambda samples, X=None: samples.squeeze(-1))
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=Y.max(),
        objective=objective,
    )

    value = acquisition(X[:2].unsqueeze(0))

    assert torch.isfinite(value).all()
