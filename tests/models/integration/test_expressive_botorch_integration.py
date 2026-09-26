"""Cross-model BoTorch integration tests for expressive surrogate models."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler

from robotorchan.models import InfiniteWidthBNNGP, JointEncoderGP, SpectralMixtureGP
from robotorchan.models.expressive.deep_gp import SingleTaskDeepGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1101)
    X = torch.rand(14, 2, dtype=torch.double)
    Y = torch.sin(4.0 * X[:, :1]) + 0.3 * X[:, 1:2]
    return X, Y


def _exact_models(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> list[tuple[str, torch.nn.Module]]:
    return [
        (
            "dkl",
            JointEncoderGP(
                X,
                Y,
                latent_dim=2,
                hidden_dims=(6,),
                random_state=11,
            ),
        ),
        ("ibnn", InfiniteWidthBNNGP(X, Y, depth=2)),
        ("spectral_mixture", SpectralMixtureGP(X, Y, num_mixtures=2)),
    ]


def test_exact_expressive_models_share_mc_acquisition_contract() -> None:
    X, Y = _data()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([8]))
    candidates = X[:2].unsqueeze(0)

    for name, model in _exact_models(X, Y):
        model.eval()
        model.likelihood.eval()
        acquisitions = [
            qLogExpectedImprovement(model=model, best_f=Y.max(), sampler=sampler),
            qLogNoisyExpectedImprovement(model=model, X_baseline=X, sampler=sampler),
            qUpperConfidenceBound(model=model, beta=0.2, sampler=sampler),
        ]
        for acquisition in acquisitions:
            value = acquisition(candidates)
            assert value.shape == torch.Size([1]), name
            assert torch.isfinite(value).all(), name


def test_exact_expressive_models_preserve_candidate_gradients() -> None:
    X, Y = _data()

    for name, model in _exact_models(X, Y):
        model.eval()
        model.likelihood.eval()
        candidate = X[:2].clone().requires_grad_(True)
        posterior = model.posterior(candidate)
        posterior.mean.sum().backward()

        assert candidate.grad is not None, name
        assert torch.isfinite(candidate.grad).all(), name


def test_exact_expressive_models_optimize_in_original_input_space() -> None:
    X, Y = _data()
    bounds = torch.stack((torch.zeros(2, dtype=X.dtype), torch.ones(2, dtype=X.dtype)))

    for name, model in _exact_models(X, Y):
        model.eval()
        model.likelihood.eval()
        acquisition = qUpperConfidenceBound(
            model=model,
            beta=0.2,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        )
        candidate, value = optimize_acqf(
            acquisition,
            bounds=bounds,
            q=1,
            num_restarts=2,
            raw_samples=12,
            options={"maxiter": 12},
        )

        assert candidate.shape == (1, 2), name
        assert torch.isfinite(candidate).all(), name
        assert torch.isfinite(value).all(), name


def test_deep_gp_shares_mc_acquisition_and_original_space_contract() -> None:
    X, Y = _data()
    model = SingleTaskDeepGP(
        X,
        Y,
        hidden_dims=(3,),
        num_inducing=5,
        posterior_samples=24,
        random_state=11,
    )
    model.eval()
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

    bounds = torch.stack((torch.zeros(2, dtype=X.dtype), torch.ones(2, dtype=X.dtype)))
    candidate, value = optimize_acqf(
        acquisitions[-1],
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=12,
        options={"maxiter": 12},
    )
    assert candidate.shape == (1, 2)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
