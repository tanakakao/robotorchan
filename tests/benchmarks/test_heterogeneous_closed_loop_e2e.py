"""Phase 13: deterministic closed-loop Bayesian optimization smoke tests."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.composition import make_qei_acquisition, make_qnei_acquisition
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics import ProblemSemantics, RegressionObjective


@pytest.mark.parametrize("kind", ["qei", "qnei"])
def test_closed_loop_retrains_and_appends_observations(kind: str) -> None:
    """Repeat fitting, acquisition, candidate selection, and noisy observation."""
    seed = 1313
    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(16, 3, generator=generator, dtype=torch.double)
    train_Y = observe(train_X, seed=seed + 100).strength.unsqueeze(-1)
    initial_X = train_X.clone()
    initial_Y = train_Y.clone()
    bounds = torch.stack((torch.zeros(3, dtype=torch.double), torch.ones(3, dtype=torch.double)))
    semantics = ProblemSemantics(objectives=(RegressionObjective(0),))
    history: list[tuple[torch.Tensor, torch.Tensor]] = []

    for iteration in range(3):
        regression = SingleTaskGP(
            train_X,
            train_Y,
            train_Yvar=torch.full_like(train_Y, 0.02**2),
        )
        fit_gpytorch_mll(regression.make_mll())
        regression.eval()
        model = HeterogeneousModel(regression)
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([16]),
            seed=seed + iteration,
        )
        if kind == "qei":
            acquisition = make_qei_acquisition(
                model,
                semantics,
                best_f=train_Y.max().item(),
                sampler=sampler,
            )
        else:
            acquisition = make_qnei_acquisition(
                model,
                semantics,
                X_baseline=train_X,
                sampler=sampler,
                prune_baseline=False,
                cache_root=False,
            )

        candidate, score = optimize_acqf(
            acquisition,
            bounds,
            q=1,
            optimizer="sobol",
            raw_samples=32,
            seed=seed + iteration,
        )
        assert candidate.shape == (1, 3)
        assert torch.isfinite(candidate).all()
        assert torch.isfinite(score).all()
        assert ((candidate >= 0) & (candidate <= 1)).all()
        with torch.no_grad():
            truth, _, _ = evaluate_truth(candidate)
            observed = observe(candidate, seed=seed + 200 + iteration)
        assert truth.shape == (1,)
        assert torch.isfinite(truth).all()
        new_Y = observed.strength.unsqueeze(-1)
        assert new_Y.shape == (1, 1)
        assert torch.isfinite(new_Y).all()
        history.append((candidate.clone(), new_Y.clone()))
        train_X = torch.cat((train_X, candidate), dim=0)
        train_Y = torch.cat((train_Y, new_Y), dim=0)
        assert train_X.shape == (17 + iteration, 3)
        assert train_Y.shape == (17 + iteration, 1)

    torch.testing.assert_close(train_X[:16], initial_X)
    torch.testing.assert_close(train_Y[:16], initial_Y)
    for iteration, (candidate, observation) in enumerate(history):
        torch.testing.assert_close(train_X[16 + iteration : 17 + iteration], candidate)
        torch.testing.assert_close(train_Y[16 + iteration : 17 + iteration], observation)
    assert len(history) == 3
