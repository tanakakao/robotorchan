"""Phase 12: fitted mixed continuous and classification constraints."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.classification_constraints import FeasibilityWeightedAcquisition
from robotorchan.acquisition.composition import (
    make_mixed_constrained_qei_acquisition,
    make_multiple_learned_constrained_qnei_acquisition,
)
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics import (
    ClassificationConstraint,
    ContinuousConstraint,
    ProblemSemantics,
    RegressionObjective,
)


@pytest.mark.parametrize("kind", ["qei", "qnei"])
@pytest.mark.parametrize("q", [1, 2])
def test_fitted_mixed_constraint_candidate_generation(kind: str, q: int) -> None:
    """Use continuous sample constraints and learned Pass PoF to generate candidates."""
    generator = torch.Generator().manual_seed(1212)
    train_X = torch.rand(24, 3, generator=generator, dtype=torch.double)
    observed = observe(train_X, seed=2212)
    train_Y = torch.stack((observed.strength, observed.conductivity), dim=-1)
    regression = SingleTaskGP(
        train_X,
        train_Y,
        train_Yvar=torch.full_like(train_Y, 0.02**2),
    )
    fit_gpytorch_mll(regression.make_mll())
    regression.eval()
    classifier = BinarySingleTaskGPClassifier(train_X, observed.passed, inducing_points=8)
    classifier.train()
    classifier.likelihood.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.03)
    for _ in range(12):
        optimizer.zero_grad()
        output = classifier.model(train_X)
        loss = -classifier.make_mll()(output, observed.passed.to(dtype=train_X.dtype))
        loss.backward()
        optimizer.step()
    classifier.eval()
    classifier.likelihood.eval()

    model = HeterogeneousModel(
        regression,
        classifier,
        output_names=["strength", "conductivity", "pass"],
    )
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"),),
        constraints=(
            ContinuousConstraint("conductivity", threshold=0.8),
            ContinuousConstraint("strength", threshold=1.5),
            ClassificationConstraint("pass", feasible_class=1),
        ),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=1212)
    if kind == "qei":
        acquisition = make_mixed_constrained_qei_acquisition(
            model,
            semantics,
            best_f=train_Y[:, 0].max().item(),
            sampler=sampler,
        )
    else:
        acquisition = make_multiple_learned_constrained_qnei_acquisition(
            model,
            semantics,
            X_baseline=train_X[:8],
            sampler=sampler,
            prune_baseline=False,
            cache_root=False,
        )

    assert isinstance(acquisition, FeasibilityWeightedAcquisition)
    X = torch.rand(2, q, 3, generator=torch.Generator().manual_seed(q), dtype=torch.double)
    with torch.no_grad():
        objective_value = acquisition.objective_acquisition(X)
        probability = acquisition.probability_of_feasibility(X)
        value = acquisition(X)
    if q == 1 and probability.shape == (2,):
        probability = probability.unsqueeze(-1)
    assert probability.shape == (2, q)
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    torch.testing.assert_close(value, objective_value * probability.prod(dim=-1))

    bounds = torch.stack((torch.zeros(3, dtype=torch.double), torch.ones(3, dtype=torch.double)))
    candidate, score = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="sobol",
        raw_samples=32,
        seed=1212 + q,
    )
    assert candidate.shape == (q, 3)
    assert torch.isfinite(score).all()
    assert ((candidate >= 0) & (candidate <= 1)).all()
    with torch.no_grad():
        strength, conductivity, true_pass = evaluate_truth(candidate)
    assert torch.isfinite(strength).all()
    assert torch.isfinite(conductivity).all()
    assert torch.isfinite(true_pass).all()
