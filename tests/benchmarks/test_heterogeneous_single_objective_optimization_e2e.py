"""Phase 7: fitted single-objective qEI/qNEI candidate-generation validation."""

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


@pytest.mark.parametrize("objective_index", [0, 1])
@pytest.mark.parametrize("acquisition_kind", ["qei", "qnei"])
def test_fitted_single_objective_candidate_generation(
    objective_index: int,
    acquisition_kind: str,
) -> None:
    """Fit a benchmark GP, compose an acquisition, and generate an in-bounds candidate."""
    seed = 710 + objective_index
    generator = torch.Generator().manual_seed(seed)
    train_X = torch.rand(24, 3, generator=generator, dtype=torch.double)
    observed = observe(train_X, seed=seed + 100)
    train_Y = (observed.strength, observed.conductivity)[objective_index].unsqueeze(-1)
    regression = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=torch.full_like(train_Y, 0.02**2),
    )
    fit_gpytorch_mll(regression.make_mll())
    regression.eval()
    model = HeterogeneousModel(regression)
    semantics = ProblemSemantics(objectives=(RegressionObjective(0),))
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=seed)
    if acquisition_kind == "qei":
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

    bounds = torch.stack(
        (torch.zeros(3, dtype=torch.double), torch.ones(3, dtype=torch.double))
    )
    candidate, acquisition_value = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        optimizer="sobol",
        raw_samples=64,
        seed=seed,
    )
    assert candidate.shape == (1, 3)
    assert torch.isfinite(candidate).all()
    assert ((candidate >= bounds[0]) & (candidate <= bounds[1])).all()
    assert torch.isfinite(acquisition_value).all()
    assert (acquisition_value >= 0).all()

    with torch.no_grad():
        reevaluated = acquisition(candidate.unsqueeze(0))
        truth = evaluate_truth(candidate)[objective_index]

    assert reevaluated.shape == (1,)
    assert torch.isfinite(reevaluated).all()
    assert truth.shape == (1,)
    assert torch.isfinite(truth).all()
