"""Phase 16: exact and variational regression models through the same BO pipeline."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.composition import make_qei_acquisition
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.models.standard.variational import SingleTaskVariationalGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics.objectives import RegressionObjective
from robotorchan.semantics.problem import ProblemSemantics


@pytest.mark.parametrize("model_kind", ["exact", "variational"])
@pytest.mark.parametrize("q", [1, 2])
def test_regression_family_acquisition_and_candidate_generation(model_kind: str, q: int) -> None:
    """Train each surrogate and exercise native posterior, qEI, and Sobol selection."""
    seed = 1616
    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(20, 3, generator=generator, dtype=torch.double)
    train_Y = observe(train_X, seed=seed + 100).strength.unsqueeze(-1)
    if model_kind == "exact":
        regression = SingleTaskGP(
            train_X,
            train_Y,
            train_Yvar=torch.full_like(train_Y, 0.02**2),
        )
        fit_gpytorch_mll(regression.make_mll())
    else:
        regression = SingleTaskVariationalGP(
            train_X,
            train_Y,
            inducing_points=8,
        )
        regression.train()
        regression.likelihood.train()
        optimizer = torch.optim.Adam(regression.parameters(), lr=0.03)
        mll = regression.make_mll()
        for _ in range(20):
            optimizer.zero_grad()
            latent = regression.model(train_X)
            loss = -mll(latent, train_Y.squeeze(-1))
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
    regression.eval()
    regression.likelihood.eval()

    torch.testing.assert_close(regression.raw_train_X, train_X)
    assert regression.raw_train_Y is not None
    torch.testing.assert_close(regression.raw_train_Y, train_Y)
    with torch.no_grad():
        posterior = regression.posterior(train_X[:3])
    assert posterior.mean.shape == (3, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

    model = HeterogeneousModel(regression, output_names=["strength"])
    semantics = ProblemSemantics(objectives=(RegressionObjective("strength"),))
    acquisition = make_qei_acquisition(
        model,
        semantics,
        best_f=train_Y.max().item(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=seed),
    )
    X = torch.rand(2, q, 3, generator=generator, dtype=torch.double)
    with torch.no_grad():
        values = acquisition(X)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()

    bounds = torch.stack((torch.zeros(3, dtype=torch.double), torch.ones(3, dtype=torch.double)))
    candidate, score = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="sobol",
        raw_samples=32,
        seed=seed + q,
    )
    assert candidate.shape == (q, 3)
    assert torch.isfinite(score).all()
    assert ((candidate >= 0) & (candidate <= 1)).all()
    with torch.no_grad():
        strength, _, _ = evaluate_truth(candidate)
    assert strength.shape == (q,)
    assert torch.isfinite(strength).all()
