"""Phase 3: fit and validate both regression outputs of the E2E benchmark."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.standard.single_task import SingleTaskGP


@pytest.mark.parametrize("output_index", [0, 1])
def test_regression_gp_end_to_end(output_index: int) -> None:
    """Fit an exact GP and check posterior shape, uncertainty, and predictive utility."""
    dtype = torch.double
    generator = torch.Generator().manual_seed(1303)
    train_X = torch.rand(32, 3, generator=generator, dtype=dtype)
    validation_X = torch.rand(16, 3, generator=generator, dtype=dtype)
    observed = observe(train_X, seed=2026, noise_std=0.02)
    train_Y = (observed.strength, observed.conductivity)[output_index].unsqueeze(-1)
    truth = evaluate_truth(validation_X)[output_index]

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=torch.full_like(train_Y, 0.02**2),
        input_transform=Normalize(d=3),
        outcome_transform=Standardize(m=1),
    )
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert torch.equal(model.raw_train_Yvar, torch.full_like(train_Y, 0.02**2))

    fit_gpytorch_mll(model.make_mll())
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(validation_X)
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)

    assert mean.shape == (16,)
    assert variance.shape == (16,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(variance).all()
    assert (variance >= 0).all()
    assert torch.mean((mean - truth).square()) < 0.12


def test_regression_gp_batch_posterior_and_samples() -> None:
    """Preserve BoTorch batch/q/output dimensions and posterior sampling."""
    dtype = torch.double
    generator = torch.Generator().manual_seed(1313)
    train_X = torch.rand(24, 3, generator=generator, dtype=dtype)
    observed = observe(train_X, seed=2027)
    model = SingleTaskGP(
        train_X,
        observed.strength.unsqueeze(-1),
        train_Yvar=torch.full((24, 1), 0.02**2, dtype=dtype),
    )
    fit_gpytorch_mll(model.make_mll())
    model.eval()
    candidate_X = torch.rand(2, 3, 3, generator=generator, dtype=dtype)
    with torch.no_grad():
        posterior = model.posterior(candidate_X)
        samples = posterior.rsample(torch.Size([4]))

    assert posterior.mean.shape == (2, 3, 1)
    assert posterior.variance.shape == (2, 3, 1)
    assert samples.shape == (4, 2, 3, 1)
    assert torch.isfinite(samples).all()
