"""Phase 9: fitted two-objective qEHVI/qNEHVI candidate generation."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

from robotorchan.acquisition.composition import (
    make_qehvi_acquisition,
    make_qnehvi_acquisition,
)
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics import ProblemSemantics, RegressionObjective


@pytest.mark.parametrize("kind", ["qehvi", "qnehvi"])
@pytest.mark.parametrize("q", [1, 2])
def test_fitted_multiobjective_candidate_generation(kind: str, q: int) -> None:
    """Fit a shared two-output GP, compose EHVI, and generate valid candidates."""
    generator = torch.Generator().manual_seed(909)
    train_X = torch.rand(24, 3, generator=generator, dtype=torch.double)
    observed = observe(train_X, seed=1909)
    train_Y = torch.stack((observed.strength, observed.conductivity), dim=-1)
    regression = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=torch.full_like(train_Y, 0.02**2),
    )
    fit_gpytorch_mll(regression.make_mll())
    regression.eval()
    model = HeterogeneousModel(regression, output_names=["strength", "conductivity"])
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"), RegressionObjective("conductivity"))
    )
    ref_point = train_Y.min(dim=0).values - 0.2
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=909)
    if kind == "qehvi":
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
        acquisition = make_qehvi_acquisition(
            model,
            semantics,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
    else:
        acquisition = make_qnehvi_acquisition(
            model,
            semantics,
            ref_point=ref_point,
            X_baseline=train_X[:8],
            sampler=sampler,
            prune_baseline=False,
            cache_root=False,
        )

    X = torch.rand(2, q, 3, generator=torch.Generator().manual_seed(q), dtype=torch.double)
    with torch.no_grad():
        values = acquisition(X)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    assert (values >= 0).all()

    bounds = torch.stack((torch.zeros(3, dtype=torch.double), torch.ones(3, dtype=torch.double)))
    candidate, score = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="sobol",
        raw_samples=32,
        seed=909 + q,
    )
    assert candidate.shape == (q, 3)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(score).all()
    assert (score >= 0).all()
    assert ((candidate >= 0) & (candidate <= 1)).all()

    with torch.no_grad():
        strength, conductivity, _ = evaluate_truth(candidate)
    assert strength.shape == (q,)
    assert conductivity.shape == (q,)
    assert torch.isfinite(strength).all()
    assert torch.isfinite(conductivity).all()
