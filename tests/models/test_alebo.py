"""Tests for ALEBO Mahalanobis GP geometry."""

import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from gpytorch.kernels import RBFKernel, ScaleKernel

from robotorchan.models import ALEBOGP
from robotorchan.models.alebo import MahalanobisRBFKernel


def test_mahalanobis_metric_is_symmetric_positive_definite() -> None:
    kernel = MahalanobisRBFKernel(ard_num_dims=3).double()

    metric = kernel.metric
    eigenvalues = torch.linalg.eigvalsh(metric)

    torch.testing.assert_close(metric, metric.transpose(-2, -1))
    assert torch.all(eigenvalues > 0)


def test_mahalanobis_kernel_has_full_metric_parameters() -> None:
    kernel = MahalanobisRBFKernel(ard_num_dims=3).double()

    with torch.no_grad():
        kernel.raw_tril.copy_(torch.tensor([0.2, 0.5, -0.3, -0.4, 0.7, 0.1], dtype=torch.double))

    metric = kernel.metric
    assert metric.shape == (3, 3)
    assert not torch.equal(metric, torch.diag(torch.diagonal(metric)))


def test_kernel_matches_exp_of_mahalanobis_distance() -> None:
    kernel = MahalanobisRBFKernel(ard_num_dims=2).double()
    x1 = torch.tensor([[0.0, 0.0]], dtype=torch.double)
    x2 = torch.tensor([[0.4, -0.2]], dtype=torch.double)

    covariance = kernel(x1, x2).to_dense()
    delta = x1[0] - x2[0]
    expected = torch.exp(-0.5 * (delta @ kernel.metric @ delta))

    torch.testing.assert_close(covariance.squeeze(), expected)


def test_alebo_gp_uses_mahalanobis_kernel_and_common_contract() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4], [0.5, 0.1]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1].square() + train_X[:, 1:].square()

    model = ALEBOGP(train_X, train_Y)

    assert isinstance(model.covar_module, ScaleKernel)
    assert isinstance(model.covar_module.base_kernel, MahalanobisRBFKernel)
    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)
    assert model.make_mll().model is model

    posterior = model.posterior(torch.zeros(1, 2, dtype=torch.double))
    assert posterior.mean.shape == (1, 1)


def test_alebo_gp_exposes_metric_from_internal_kernel() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4], [0.5, 0.1]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1].square() + train_X[:, 1:].square()

    model = ALEBOGP(train_X, train_Y)

    torch.testing.assert_close(model.metric, model.covar_module.base_kernel.metric)


def test_alebo_gp_fit_returns_same_model(monkeypatch) -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4], [0.5, 0.1]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1].square() + train_X[:, 1:].square()
    model = ALEBOGP(train_X, train_Y)
    captured = {}

    def fake_fit(mll, **kwargs):
        captured["mll"] = mll
        captured["kwargs"] = kwargs
        return mll

    monkeypatch.setattr("robotorchan.models.alebo.fit_gpytorch_mll", fake_fit)

    fitted = model.fit(optimizer_kwargs={"options": {"maxiter": 3}})

    assert fitted is model
    assert captured["mll"].model is not model
    torch.testing.assert_close(captured["mll"].model.raw_train_X, model.raw_train_X)
    torch.testing.assert_close(captured["mll"].model.raw_train_Y, model.raw_train_Y)
    assert captured["kwargs"] == {"optimizer_kwargs": {"options": {"maxiter": 3}}}


def test_alebo_fit_warm_starts_first_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    with torch.no_grad():
        model.mean_module.constant.fill_(1.25)
    seen_means = []

    def fake_fit(mll, **kwargs):
        del kwargs
        seen_means.append(mll.model.mean_module.constant.detach().clone())
        return mll

    monkeypatch.setattr("robotorchan.models.alebo.fit_gpytorch_mll", fake_fit)

    model.fit(restarts=2)

    torch.testing.assert_close(seen_means[0], torch.full_like(seen_means[0], 1.25))
    assert len(seen_means) == 2


def test_alebo_fit_can_disable_warm_start(monkeypatch: pytest.MonkeyPatch) -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    with torch.no_grad():
        model.mean_module.constant.fill_(9.0)
    seen_means = []

    def fake_fit(mll, **kwargs):
        del kwargs
        seen_means.append(mll.model.mean_module.constant.detach().clone())
        return mll

    monkeypatch.setattr("robotorchan.models.alebo.fit_gpytorch_mll", fake_fit)

    model.fit(restarts=1, warm_start=False)

    assert not torch.equal(seen_means[0], torch.full_like(seen_means[0], 9.0))


def test_alebo_gp_rejects_non_mahalanobis_covar_for_metric_access() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y, covar_module=ScaleKernel(RBFKernel()))

    with pytest.raises(TypeError, match="MahalanobisRBFKernel"):
        _ = model.metric


def test_metric_parameter_vector_is_detached_copy() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)

    with torch.no_grad():
        model.mahalanobis_kernel.raw_tril.fill_(0.5)
    vector = model.metric_parameter_vector()

    assert vector.shape == (1,)
    assert not vector.requires_grad
    vector.zero_()
    assert not torch.equal(vector, model.mahalanobis_kernel.raw_tril)


def test_sample_metric_parameters_uses_gaussian_laplace_covariance() -> None:
    train_X = torch.tensor(
        [[0.0, 0.0], [0.2, -0.1], [-0.3, 0.4], [0.5, 0.1]],
        dtype=torch.double,
    )
    train_Y = train_X[:, :1].square() + train_X[:, 1:].square()
    model = ALEBOGP(train_X, train_Y)
    n_params = model.metric_parameter_vector().numel()
    covariance = 0.01 * torch.eye(n_params, dtype=torch.double)
    generator = torch.Generator().manual_seed(7)

    samples = model.sample_metric_parameters(5, covariance=covariance, generator=generator)

    assert samples.shape == (5, n_params)
    assert samples.dtype == torch.double


def test_sample_metric_parameters_validates_inputs() -> None:
    train_X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)

    with pytest.raises(ValueError, match="n_samples"):
        model.sample_metric_parameters(0, covariance=torch.eye(1, dtype=torch.double))
    with pytest.raises(ValueError, match="covariance must have shape"):
        model.sample_metric_parameters(2, covariance=torch.eye(2, dtype=torch.double))


def test_metric_parameter_vector_uses_only_free_lower_triangle() -> None:
    train_X = torch.zeros(3, 3, dtype=torch.double)
    train_Y = torch.zeros(3, 1, dtype=torch.double)
    model = ALEBOGP(train_X, train_Y)

    assert model.metric_parameter_vector().shape == (6,)


def test_metric_laplace_covariance_uses_negative_diagonal_hessian() -> None:
    train_X = torch.zeros(3, 2, dtype=torch.double)
    train_Y = torch.zeros(3, 1, dtype=torch.double)
    model = ALEBOGP(train_X, train_Y)

    covariance = model.metric_laplace_covariance(
        diagonal_hessian=torch.tensor([-2.0, -4.0, -5.0], dtype=torch.double)
    )

    expected = torch.diag(1 / torch.tensor([2.001, 4.001, 5.001], dtype=torch.double))
    assert torch.allclose(covariance, expected)


def test_metric_laplace_covariance_uses_reference_nugget_stabilization() -> None:
    train_X = torch.zeros(3, 2, dtype=torch.double)
    train_Y = torch.zeros(3, 1, dtype=torch.double)
    model = ALEBOGP(train_X, train_Y)

    covariance = model.metric_laplace_covariance(
        diagonal_hessian=torch.tensor([-1.0, 0.0, -2.0], dtype=torch.double)
    )

    expected = torch.diag(1 / torch.tensor([1.001, 0.001, 2.001], dtype=torch.double))
    torch.testing.assert_close(covariance, expected)


def test_moment_match_predictions_includes_between_model_uncertainty() -> None:
    means = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.double)
    variances = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.double)

    mean, variance = ALEBOGP.moment_match_predictions(means, variances)

    torch.testing.assert_close(mean, torch.tensor([1.0, 2.0], dtype=torch.double))
    torch.testing.assert_close(variance, torch.tensor([2.0, 2.0], dtype=torch.double))


def test_moment_match_predictions_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ALEBOGP.moment_match_predictions(torch.zeros(2, 1), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        ALEBOGP.moment_match_predictions(torch.zeros(1, 1), -torch.ones(1, 1))


def test_metric_diagonal_hessian_matches_autograd_curvature() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)

    diagonal = model.metric_diagonal_hessian()

    assert diagonal.shape == model.metric_parameter_vector().shape
    assert diagonal.dtype == torch.double
    assert torch.isfinite(diagonal).all()


def test_estimate_metric_laplace_covariance_uses_automatic_hessian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_X = torch.zeros(3, 2, dtype=torch.double)
    train_Y = torch.zeros(3, 1, dtype=torch.double)
    model = ALEBOGP(train_X, train_Y)
    diagonal = torch.tensor([-2.0, -4.0, -5.0], dtype=torch.double)
    monkeypatch.setattr(model, "metric_diagonal_hessian", lambda: diagonal)

    covariance = model.estimate_metric_laplace_covariance()

    expected = torch.diag(1 / torch.tensor([2.001, 4.001, 5.001], dtype=torch.double))
    torch.testing.assert_close(covariance, expected)


def test_metric_sample_predictions_restore_fitted_metric() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    original = model.metric_parameter_vector()
    samples = torch.stack([original - 0.1, original + 0.1])

    means, covariances = model.metric_sample_predictions(train_X, metric_samples=samples)

    assert means.shape[0] == 2
    assert covariances.shape == (2, 3, 3)
    torch.testing.assert_close(model.metric_parameter_vector(), original)


def test_marginal_metric_moments_include_metric_uncertainty() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    covariance = torch.eye(1, dtype=torch.double) * 0.01
    generator = torch.Generator().manual_seed(11)

    mean, predictive_covariance = model.marginal_metric_moments(
        train_X,
        n_metric_samples=3,
        covariance=covariance,
        generator=generator,
    )

    assert mean.shape == train_Y.shape
    assert predictive_covariance.shape == (3, 3)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(predictive_covariance).all()
    torch.testing.assert_close(predictive_covariance, predictive_covariance.transpose(-2, -1))


def test_metric_sample_predictions_validates_sample_shape() -> None:
    train_X = torch.zeros(3, 2, dtype=torch.double)
    train_Y = torch.zeros(3, 1, dtype=torch.double)
    model = ALEBOGP(train_X, train_Y)

    with pytest.raises(ValueError, match="metric_samples must have shape"):
        model.metric_sample_predictions(train_X, metric_samples=torch.zeros(2, 2))


def test_marginal_metric_posterior_is_botorch_compatible() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    covariance = torch.eye(1, dtype=torch.double) * 0.01
    generator = torch.Generator().manual_seed(17)

    posterior = model.marginal_metric_posterior(
        train_X,
        n_metric_samples=3,
        covariance=covariance,
        generator=generator,
    )

    assert posterior.mean.shape == train_Y.shape
    assert posterior.variance.shape == train_Y.shape
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    samples = posterior.rsample(torch.Size([4]))
    assert samples.shape == (4, *train_Y.shape)


def test_acquisition_model_uses_metric_marginal_posterior() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    covariance = torch.eye(1, dtype=torch.double) * 0.01
    acquisition_model = model.acquisition_model(
        n_metric_samples=3,
        covariance=covariance,
        generator=torch.Generator().manual_seed(23),
    )

    posterior = acquisition_model.posterior(train_X)

    assert posterior.mean.shape == train_Y.shape
    assert posterior.variance.shape == train_Y.shape


def test_log_ei_accepts_alebo_metric_marginal_model() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    covariance = torch.eye(1, dtype=torch.double) * 0.01
    acquisition_model = model.acquisition_model(
        n_metric_samples=3,
        covariance=covariance,
        generator=torch.Generator().manual_seed(29),
    )
    acquisition = LogExpectedImprovement(model=acquisition_model, best_f=train_Y.max())

    value = acquisition(torch.tensor([[0.25]], dtype=torch.double))

    assert value.numel() == 1
    assert torch.isfinite(value).all()


def test_marginal_metric_posterior_preserves_cross_point_covariance() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    covariance = torch.eye(1, dtype=torch.double) * 0.01

    posterior = model.marginal_metric_posterior(
        train_X,
        n_metric_samples=3,
        covariance=covariance,
        generator=torch.Generator().manual_seed(31),
    )
    predictive_covariance = posterior.distribution.covariance_matrix

    assert predictive_covariance.shape == (3, 3)
    off_diagonal = predictive_covariance - torch.diag_embed(torch.diagonal(predictive_covariance))
    assert torch.any(off_diagonal.abs() > 0)


def test_acquisition_model_reuses_fixed_metric_samples() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    covariance = torch.eye(1, dtype=torch.double) * 0.01
    acquisition_model = model.acquisition_model(
        n_metric_samples=3,
        covariance=covariance,
        generator=torch.Generator().manual_seed(41),
    )
    test_X = torch.tensor([[-0.25], [0.25]], dtype=torch.double)

    first = acquisition_model.posterior(test_X)
    second = acquisition_model.posterior(test_X)

    torch.testing.assert_close(first.mean, second.mean)
    torch.testing.assert_close(
        first.distribution.covariance_matrix,
        second.distribution.covariance_matrix,
    )
    assert acquisition_model.metric_samples.shape == (3, 1)


def test_metric_samples_include_map_as_first_sample() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)
    covariance = torch.eye(1, dtype=torch.double) * 0.01
    map_metric = model.metric_parameter_vector().detach().clone()

    samples = model.sample_metric_parameters(
        4,
        covariance=covariance,
        generator=torch.Generator().manual_seed(43),
    )

    torch.testing.assert_close(samples[0], map_metric)
    assert samples.shape == (4, 1)


def test_projection_initializes_nontrivial_alebo_metric() -> None:
    projection = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]],
        dtype=torch.double,
    )
    train_X = torch.tensor([[-0.5, 0.0], [0.0, 0.25], [0.5, -0.25]], dtype=torch.double)
    train_Y = train_X.square().sum(dim=-1, keepdim=True)

    model = ALEBOGP(train_X, train_Y, projection=projection)

    assert model.metric.shape == (2, 2)
    assert torch.isfinite(model.metric).all()
    assert torch.all(torch.linalg.eigvalsh(model.metric) > 0)
    assert not torch.allclose(model.metric, torch.eye(2, dtype=torch.double))


def test_fit_validates_alebo_map_restarts() -> None:
    train_X = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.double)
    train_Y = train_X.square()
    model = ALEBOGP(train_X, train_Y)

    with pytest.raises(ValueError, match="restarts must be positive"):
        model.fit(restarts=0)


def test_alebo_kernel_retains_projection_for_reference_restarts() -> None:
    projection = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]],
        dtype=torch.double,
    )
    train_X = torch.zeros(3, 2, dtype=torch.double)
    train_Y = torch.zeros(3, 1, dtype=torch.double)
    model = ALEBOGP(train_X, train_Y, projection=projection)

    torch.testing.assert_close(model.mahalanobis_kernel.projection, projection)
