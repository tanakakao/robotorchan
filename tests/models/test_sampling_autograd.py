"""Cross-model autograd contracts across posterior sampling."""

import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import RandomForestSurrogate, SingleTaskDeepGP, SingleTaskGP


def test_standard_gp_sample_gradient_reaches_candidate() -> None:
    train_x = torch.rand(12, 2, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 3.0)
    model = SingleTaskGP(train_x, train_y)
    candidate = torch.rand(3, 2, dtype=torch.double, requires_grad=True)

    samples = SobolQMCNormalSampler(torch.Size([16]), seed=123)(model.posterior(candidate))
    gradient = torch.autograd.grad(samples.mean(), candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()


def test_deep_gp_sample_gradient_reaches_candidate_through_stored_trajectory() -> None:
    torch.manual_seed(17)
    train_x = torch.rand(12, 2, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 3.0)
    model = SingleTaskDeepGP(
        train_x,
        train_y,
        hidden_dims=(3,),
        num_inducing=5,
        posterior_samples=16,
    )
    model.eval()
    candidate = torch.rand(3, 2, dtype=torch.double, requires_grad=True)

    posterior = model.posterior(candidate, num_samples=16)
    samples = SobolQMCNormalSampler(torch.Size([8]), seed=456)(posterior)
    gradient = torch.autograd.grad(samples.mean(), candidate)[0]

    assert gradient.shape == candidate.shape
    assert torch.isfinite(gradient).all()


def test_tree_ensemble_posterior_is_non_differentiable_at_model_boundary() -> None:
    train_x = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 3.0)
    model = RandomForestSurrogate(
        train_x,
        train_y,
        n_estimators=8,
        random_state=0,
    )
    model.fit()
    candidate = torch.tensor(
        [[0.25], [0.75]],
        dtype=torch.double,
        requires_grad=True,
    )

    posterior = model.posterior(candidate)

    assert not posterior.values.requires_grad
    assert posterior.mean.grad_fn is None
