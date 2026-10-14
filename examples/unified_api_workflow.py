"""Runnable example: unified GP API and native BoTorch acquisition components."""

import torch
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import LinearMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP
from robotorchan.optim import optimize_acqf


def main() -> None:
    """Fit a multi-output exact GP and propose two bounded candidates."""
    train_X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (torch.sin(3 * train_X), torch.cos(3 * train_X)),
        dim=-1,
    )
    model = SingleTaskGP(train_X, train_Y)
    assert model.supports_fit
    model.fit(optimizer_kwargs={"options": {"maxiter": 2}})

    test_X = train_X[:3]
    posterior = model.posterior(test_X)
    assert posterior.mean.shape == (3, 2)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=110)
    objective = LinearMCObjective(weights=torch.tensor([0.7, 0.3], dtype=torch.double))
    acquisition = qSimpleRegret(model=model, sampler=sampler, objective=objective)

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, score = optimize_acqf(
        acquisition,
        bounds,
        q=2,
        optimizer="sobol",
        raw_samples=32,
        seed=110,
    )
    assert candidates.shape == (2, 1)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(score).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
    print("posterior mean:", posterior.mean)
    print("proposed candidates:", candidates)
    print("acquisition value:", score)


if __name__ == "__main__":
    main()
