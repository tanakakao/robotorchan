"""Native temporal kernel composition and exact-GP integration."""

import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel, ScaleKernel

from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.temporal import make_temporal_kernel


@pytest.mark.parametrize(
    "kind",
    [
        "rbf",
        "matern",
        "exponential",
        "rq",
        "periodic",
        "locally_periodic",
        "linear",
        "spectral_mixture",
    ],
)
def test_covariance_psd_and_active_dimension(kind):
    x = torch.tensor([[0.0, 0.0], [9.0, 0.1], [1.0, 0.4], [2.0, 0.9]]).double()
    kernel = make_temporal_kernel(kind, time_dim=1).double()
    cov = kernel(x).to_dense()
    torch.testing.assert_close(cov, cov.T)
    assert torch.linalg.eigvalsh(cov).min() >= -1e-8
    changed = x.clone()
    changed[:, 0] += torch.arange(4).double()
    torch.testing.assert_close(kernel(changed).to_dense(), cov)


def test_periodic_and_local_periodic():
    t = torch.tensor([[0.0], [2.0], [4.0]], dtype=torch.double)
    periodic = make_temporal_kernel("periodic", period_length=2.0).double()
    local = make_temporal_kernel(
        "locally_periodic", period_length=2.0, local_lengthscale=1.0
    ).double()
    cov = periodic(t).to_dense()
    torch.testing.assert_close(cov[0, 0], cov[0, 2])
    local_cov = local(t).to_dense()
    assert local_cov[0, 0] > local_cov[0, 1] > local_cov[0, 2]


def test_exponential_matches_matern_half():
    t = torch.tensor([[0.0], [0.3], [1.0]], dtype=torch.double)
    exponential = make_temporal_kernel("exponential", lengthscale=0.2).double()
    matern = make_temporal_kernel("matern", nu=0.5, lengthscale=0.2).double()
    torch.testing.assert_close(exponential(t).to_dense(), matern(t).to_dense())


def test_fit_forecast_and_fixed_time_candidate():
    t = torch.tensor([0.0, 0.08, 0.21, 0.33, 0.51, 0.68, 0.82, 1.0]).double()
    x = torch.linspace(0, 1, len(t), dtype=torch.double)
    train_x = torch.stack((t, x), dim=-1)
    y = (torch.sin(2 * torch.pi * t) + x).unsqueeze(-1)
    temporal = make_temporal_kernel("locally_periodic", period_length=1.0)
    control = ScaleKernel(RBFKernel(active_dims=(1,)))
    model = SingleTaskGP(train_x, y, covar_module=(temporal + control).double())
    model.fit(optimizer_kwargs={"options": {"maxiter": 30}})
    future = torch.tensor([[1.1, 0.2], [1.2, 0.8]], dtype=torch.double)
    posterior = model.posterior(future)
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert posterior.rsample(torch.Size([4])).shape == (4, 2, 1)
    acqf = LogExpectedImprovement(model, best_f=y.max())
    candidate, value = optimize_acqf(
        acqf,
        bounds=torch.tensor([[1.1, 0.0], [1.2, 1.0]], dtype=torch.double),
        q=1,
        num_restarts=2,
        raw_samples=16,
        fixed_features={0: 1.15},
    )
    torch.testing.assert_close(candidate[0, 0], candidate.new_tensor(1.15))
    assert 0 <= candidate[0, 1] <= 1
    assert torch.isfinite(value).all()


def test_invalid_parameters():
    with pytest.raises(ValueError, match="positive"):
        make_temporal_kernel("periodic", period_length=0)
    with pytest.raises(ValueError, match="periodic"):
        make_temporal_kernel("rbf", period_length=1.0)
    with pytest.raises(ValueError, match="column"):
        make_temporal_kernel("rbf", time_dim=-1)
