"""Tests for the spectral-mixture Gaussian process model."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from gpytorch.kernels import ScaleKernel, SpectralMixtureKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models.expressive.spectral_mixture import (
    MixedSpectralMixtureGP,
    SpectralMixtureGP,
    SpectralMixtureKroneckerMultiTaskGP,
    SpectralMixtureMultiTaskGP,
)


def _periodic_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(2.0 * torch.pi * 3.0 * X) + 0.25 * torch.sin(2.0 * torch.pi * 7.0 * X)
    return X, Y


def test_spectral_mixture_gp_initializes_finite_kernel_parameters() -> None:
    X, Y = _periodic_data()
    model = SpectralMixtureGP(X, Y, num_mixtures=3)

    assert isinstance(model.covar_module, ScaleKernel)
    kernel = model.covar_module.base_kernel
    assert isinstance(kernel, SpectralMixtureKernel)
    assert kernel.num_mixtures == 3
    assert torch.isfinite(kernel.mixture_weights).all()
    assert torch.isfinite(kernel.mixture_means).all()
    assert torch.isfinite(kernel.mixture_scales).all()


def test_spectral_mixture_gp_retains_data_and_mll_contract() -> None:
    X, Y = _periodic_data()
    model = SpectralMixtureGP(X, Y)

    assert torch.equal(model.raw_train_X, X)
    assert torch.equal(model.raw_train_Y, Y)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_spectral_mixture_gp_supports_posterior_and_qlogei() -> None:
    X, Y = _periodic_data()
    model = SpectralMixtureGP(X, Y, num_mixtures=2)
    model.eval()
    model.likelihood.eval()

    candidates = torch.tensor([[0.15], [0.55]], dtype=torch.double)
    posterior = model.posterior(candidates)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

    acquisition = qLogExpectedImprovement(model=model, best_f=Y.max())
    values = acquisition(candidates.unsqueeze(-2))
    assert torch.isfinite(values).all()


def test_spectral_mixture_gp_optimize_acqf_uses_original_space() -> None:
    X, Y = _periodic_data()
    model = SpectralMixtureGP(X, Y, num_mixtures=2)
    model.eval()
    model.likelihood.eval()

    acquisition = qLogExpectedImprovement(model=model, best_f=Y.max())
    bounds = torch.tensor([[0.05], [0.95]], dtype=torch.double)
    candidate, _ = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
    )
    assert candidate.shape == (1, 1)
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])


def test_spectral_mixture_gp_validates_configuration() -> None:
    X, Y = _periodic_data()

    for kwargs in ({"num_mixtures": 0}, {"initialization": "invalid"}):
        try:
            SpectralMixtureGP(X, Y, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {kwargs}.")


def _multitask_periodic_data() -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)
    tasks = torch.arange(24, dtype=torch.double).remainder(2).unsqueeze(-1)
    X = torch.cat((base, tasks), dim=-1)
    Y = torch.sin(2.0 * torch.pi * 3.0 * base) + 0.3 * tasks
    return X, Y


def test_spectral_mixture_multitask_preserves_task_structure() -> None:
    X, Y = _multitask_periodic_data()
    model = SpectralMixtureMultiTaskGP(X, Y, task_feature=1, num_mixtures=2)

    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    data_kernel = model.covar_module.kernels[0]
    assert data_kernel.base_kernel.active_dims.tolist() == [0]
    assert data_kernel.base_kernel.num_mixtures == 2


def test_spectral_mixture_multitask_posterior_is_finite() -> None:
    X, Y = _multitask_periodic_data()
    model = SpectralMixtureMultiTaskGP(X, Y, task_feature=1, num_mixtures=2)
    model.eval()

    posterior = model.posterior(X[:4])

    assert posterior.mean.shape == (4, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_spectral_mixture_multitask_qlogei_runs() -> None:
    X, Y = _multitask_periodic_data()
    model = SpectralMixtureMultiTaskGP(X, Y, task_feature=1, num_mixtures=2)
    model.eval()
    objective = GenericMCObjective(lambda samples, X=None: samples.squeeze(-1))
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=Y.max(),
        objective=objective,
    )

    value = acquisition(X[:2].unsqueeze(0))

    assert torch.isfinite(value).all()


def test_mixed_spectral_mixture_supports_posterior() -> None:
    X = torch.tensor(
        [[0.0, 0.0], [0.2, 1.0], [0.4, 0.0], [0.6, 1.0], [0.8, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    Y = torch.sin(2.0 * torch.pi * X[:, :1]) + 0.2 * X[:, 1:2]
    model = MixedSpectralMixtureGP(X, Y, cat_dims=[1], num_mixtures=2)
    model.eval()

    posterior = model.posterior(X[:2])

    assert model.cat_dims == (1,)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def _kronecker_periodic_data() -> tuple[torch.Tensor, torch.Tensor]:
    X, base = _periodic_data()
    other = 0.6 * base + 0.2 * torch.cos(2.0 * torch.pi * X)
    return X, torch.cat((base, other), dim=-1)


def test_spectral_mixture_kronecker_preserves_block_design_and_kernel() -> None:
    X, Y = _kronecker_periodic_data()
    model = SpectralMixtureKroneckerMultiTaskGP(X, Y, num_mixtures=2)
    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    data_kernel = model.covar_module.data_covar_module
    assert isinstance(data_kernel, ScaleKernel)
    kernel = data_kernel.base_kernel
    assert isinstance(kernel, SpectralMixtureKernel)
    assert kernel.num_mixtures == 2
    assert torch.isfinite(kernel.mixture_weights).all()
    assert torch.isfinite(kernel.mixture_means).all()
    assert torch.isfinite(kernel.mixture_scales).all()


def test_spectral_mixture_kronecker_posterior_sampling_is_finite() -> None:
    X, Y = _kronecker_periodic_data()
    model = SpectralMixtureKroneckerMultiTaskGP(X, Y, num_mixtures=2)
    model.eval()
    model.likelihood.eval()
    posterior = model.posterior(X[:3])
    samples = posterior.rsample(torch.Size([4]))
    assert posterior.mean.shape == (3, 2)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert samples.shape == (4, 3, 2)
    assert torch.isfinite(samples).all()


def test_spectral_mixture_kronecker_scalarized_mc_and_optimizer_run() -> None:
    X, Y = _kronecker_periodic_data()
    model = SpectralMixtureKroneckerMultiTaskGP(X, Y, num_mixtures=2)
    model.eval()
    model.likelihood.eval()
    objective = GenericMCObjective(lambda samples, X=None: samples.mean(dim=-1))
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=Y.mean(dim=-1).max(),
        objective=objective,
    )
    assert torch.isfinite(acquisition(X[:2].unsqueeze(0))).all()
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidate, value = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 12},
    )
    assert candidate.shape == (1, 1)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
