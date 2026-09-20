"""BoTorch integration tests for SingleTaskDeepGP."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.sampling.stochastic_samplers import StochasticSampler

from robotorchan.models.deep_gp import SingleTaskDeepGP


def _model() -> tuple[SingleTaskDeepGP, torch.Tensor, torch.Tensor]:
    torch.manual_seed(19)
    X = torch.rand(14, 2, dtype=torch.double)
    Y = torch.sin(5.0 * X[:, :1]) + 0.2 * X[:, 1:2]
    model = SingleTaskDeepGP(
        X,
        Y,
        hidden_dims=(3,),
        num_inducing=5,
        posterior_samples=24,
        random_state=11,
    )
    model.eval()
    return model, X, Y


def test_deep_gp_posterior_has_botorch_shape_and_gradients() -> None:
    model, X, _ = _model()
    candidate = X[:3].clone().requires_grad_(True)

    posterior = model.posterior(candidate)
    assert posterior.mean.shape == (3, 1)
    assert posterior.variance.shape == (3, 1)

    posterior.mean.sum().backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_deep_gp_posterior_rsample_shape() -> None:
    model, X, _ = _model()
    posterior = model.posterior(X[:3])

    samples = posterior.rsample(torch.Size([7]))

    assert samples.shape == (7, 3, 1)
    assert torch.isfinite(samples).all()


def test_deep_gp_qlogei_and_qucb_are_finite() -> None:
    model, X, Y = _model()
    sampler = StochasticSampler(sample_shape=torch.Size([8]))
    candidates = X[:2].unsqueeze(0)

    acquisitions = [
        qLogExpectedImprovement(model=model, best_f=Y.max(), sampler=sampler),
        qUpperConfidenceBound(model=model, beta=0.2, sampler=sampler),
    ]

    for acquisition in acquisitions:
        value = acquisition(candidates)
        assert value.shape == torch.Size([1])
        assert torch.isfinite(value).all()


def test_deep_gp_optimize_acqf_runs_in_original_space() -> None:
    model, X, _ = _model()
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=StochasticSampler(sample_shape=torch.Size([8])),
    )
    bounds = torch.stack((torch.zeros(2, dtype=X.dtype), torch.ones(2, dtype=X.dtype)))

    candidate, value = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=12,
        options={"maxiter": 12},
    )

    assert candidate.shape == (1, 2)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_deep_gp_single_observation_standardization_is_finite() -> None:
    X = torch.tensor([[0.25, 0.75]], dtype=torch.double)
    Y = torch.tensor([[1.0]], dtype=torch.double)
    model = SingleTaskDeepGP(X, Y, hidden_dims=(2,), num_inducing=3)

    assert torch.isfinite(model.input_scale).all()
    assert torch.isfinite(model.training_loss(num_likelihood_samples=2))
