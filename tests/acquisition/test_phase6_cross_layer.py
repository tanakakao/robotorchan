"""Phase 6 cross-layer contracts using unwrapped BoTorch components."""

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf


def _components() -> tuple[SingleTaskGP, SobolQMCNormalSampler, LinearMCObjective]:
    X = torch.linspace(0.1, 0.9, 9, dtype=torch.double).unsqueeze(-1)
    Y = torch.cat((torch.sin(3 * X), torch.cos(3 * X)), dim=-1)
    model = SingleTaskGP(X, Y)
    model.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=2606)
    objective = LinearMCObjective(weights=torch.tensor([0.7, 0.3], dtype=X.dtype))
    return model, sampler, objective


@pytest.mark.parametrize("q", [1, 2])
def test_posterior_sampler_objective_acquisition_shapes_and_gradients(q: int) -> None:
    """Connect multi-output GP to native sampling and scalar MC optimization."""
    model, sampler, objective = _components()
    X = torch.linspace(0.2, 0.8, 2 * q, dtype=torch.double)
    X = X.reshape(2, q, 1).requires_grad_()

    posterior = model.posterior(X)
    assert posterior.mean.shape == (2, q, 2)
    samples = sampler(posterior)
    assert samples.shape == (16, 2, q, 2)

    values = objective(samples, X=X)
    assert values.shape == (16, 2, q)
    assert torch.isfinite(values).all()

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=0.0,
        sampler=sampler,
        objective=objective,
    )
    assert acquisition.__class__.__module__.startswith("botorch.")
    scores = acquisition(X)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    scores.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()


@pytest.mark.parametrize("optimizer", ["botorch", "sobol"])
def test_native_objective_flows_through_robotorchan_optimizer(optimizer: str) -> None:
    """Verify candidate search does not replace the posterior or MC objective."""
    model, sampler, objective = _components()
    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=0.0,
        sampler=sampler,
        objective=objective,
    )
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    kwargs = (
        {"num_restarts": 2, "raw_samples": 16, "options": {"maxiter": 5}}
        if optimizer == "botorch"
        else {"raw_samples": 16, "seed": 2606}
    )
    candidates, score = optimize_acqf(
        acquisition,
        bounds,
        q=2,
        optimizer=optimizer,
        **kwargs,
    )
    assert candidates.shape == (2, 1)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(score).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
