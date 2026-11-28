"""Temporal covariance integration with mixed, batched and multi-output GP models."""

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models.standard.single_task import MixedSingleTaskGP, SingleTaskGP
from robotorchan.temporal import make_temporal_kernel


def test_batched_and_independent_multioutput():
    t = torch.tensor([0.0, 0.13, 0.32, 0.55, 0.78, 1.0], dtype=torch.double)
    x = t[:, None]
    y = torch.stack((torch.sin(6 * t), torch.cos(6 * t)), dim=-1)
    kernel = make_temporal_kernel("matern", batch_shape=torch.Size([2])).double()
    model = SingleTaskGP(x, y, covar_module=kernel)
    model.fit(optimizer_kwargs={"options": {"maxiter": 20}})
    test = torch.tensor([[[1.1], [1.2]], [[1.3], [1.4]]], dtype=torch.double)
    posterior = model.posterior(test)
    assert posterior.mean.shape == (2, 2, 2)
    assert posterior.variance.shape == (2, 2, 2)
    assert torch.isfinite(posterior.mean).all()
    assert posterior.rsample(torch.Size([4])).shape == (4, 2, 2, 2)


def test_temporal_kernel_model_batch():
    t = torch.linspace(0, 1, 6, dtype=torch.double)
    x = t.expand(2, -1).unsqueeze(-1)
    y = torch.stack((torch.sin(t), torch.cos(t))).unsqueeze(-1)
    kernel = make_temporal_kernel("periodic", batch_shape=torch.Size([2])).double()
    model = SingleTaskGP(x, y, covar_module=kernel)
    posterior = model.posterior(torch.tensor([[1.1], [1.2]], dtype=torch.double))
    assert posterior.mean.shape == (2, 2, 1)
    assert torch.isfinite(posterior.variance).all()


def test_mixed_temporal_q_batch():
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 8, dtype=torch.double)
    u = torch.tensor([0.0, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 1.0], dtype=t.dtype)
    cat = torch.arange(8, dtype=t.dtype) % 2
    x = torch.stack((t, u, cat), dim=-1)
    y = (torch.sin(6 * t) + u + 0.2 * cat).unsqueeze(-1)

    def continuous_kernel(batch_shape, num_dims, active_dims):
        assert num_dims == len(active_dims)
        time = make_temporal_kernel("locally_periodic", time_dim=0, batch_shape=batch_shape)
        control = make_temporal_kernel("rbf", time_dim=1, batch_shape=batch_shape)
        return time * control

    model = MixedSingleTaskGP(x, y, cat_dims=[2], cont_kernel_factory=continuous_kernel)
    model.fit(optimizer_kwargs={"options": {"maxiter": 20}})
    acq = qLogExpectedImprovement(
        model, best_f=y.max(), sampler=SobolQMCNormalSampler(torch.Size([32]), seed=0)
    )
    bounds = torch.tensor([[1.0, 0.0, 0.0], [1.2, 1.0, 1.0]], dtype=t.dtype)
    # Optimize each category separately, preserving a fixed future time.
    candidates = []
    scores = []
    for category in (0.0, 1.0):
        candidate, value = optimize_acqf(
            acq,
            bounds=bounds,
            q=2,
            num_restarts=2,
            raw_samples=16,
            fixed_features={0: 1.1, 2: category},
            options={"maxiter": 30},
        )
        candidates.append(candidate)
        scores.append(value)
    candidate = candidates[torch.stack(scores).argmax().item()]
    assert candidate.shape == (2, 3)
    torch.testing.assert_close(candidate[:, 0], candidate.new_full((2,), 1.1))
    assert ((candidate[:, 1] >= 0) & (candidate[:, 1] <= 1)).all()
    assert ((candidate[:, 2] == 0) | (candidate[:, 2] == 1)).all()
    assert torch.isfinite(torch.stack(scores)).all()
