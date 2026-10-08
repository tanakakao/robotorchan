"""Phase 10: fitted multiobjective optimization with binary Pass feasibility."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

from robotorchan.acquisition.classification_constraints import FeasibilityWeightedAcquisition
from robotorchan.acquisition.composition import (
    make_constrained_qehvi_acquisition,
    make_qnehvi_acquisition,
    resolve_acquisition_composition,
)
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics import ClassificationConstraint, ProblemSemantics, RegressionObjective


@pytest.mark.parametrize("kind", ["qehvi", "qnehvi"])
@pytest.mark.parametrize("q", [1, 2])
def test_classification_constrained_multiobjective_candidates(kind: str, q: int) -> None:
    """Fit both model families and optimize PoF-weighted multiobjective acquisitions."""
    generator = torch.Generator().manual_seed(1010)
    train_X = torch.rand(24, 3, generator=generator, dtype=torch.double)
    observed = observe(train_X, seed=2010)
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
    objectives = (
        RegressionObjective("strength"),
        RegressionObjective("conductivity"),
    )
    semantics = ProblemSemantics(
        objectives=objectives,
        constraints=(ClassificationConstraint("pass", feasible_class=1),),
    )
    ref_point = train_Y.min(dim=0).values - 0.2
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=1010)
    if kind == "qehvi":
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
        acquisition = make_constrained_qehvi_acquisition(
            model,
            semantics,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
    else:
        unconstrained = make_qnehvi_acquisition(
            model,
            ProblemSemantics(objectives=objectives),
            ref_point=ref_point,
            X_baseline=train_X[:8],
            sampler=sampler,
            prune_baseline=False,
            cache_root=False,
        )
        feasibility = resolve_acquisition_composition(model, semantics).feasibility[0]
        acquisition = FeasibilityWeightedAcquisition(
            unconstrained,
            feasibility.representation.probability,
        )

    assert isinstance(acquisition, FeasibilityWeightedAcquisition)
    X = torch.rand(2, q, 3, generator=torch.Generator().manual_seed(q), dtype=torch.double)
    with torch.no_grad():
        objective_values = acquisition.objective_acquisition(X)
        probability = acquisition.probability_of_feasibility(X)
        weighted = acquisition(X)
    if q == 1 and probability.shape == (2,):
        probability = probability.unsqueeze(-1)
    assert probability.shape == (2, q)
    assert weighted.shape == (2,)
    assert torch.isfinite(weighted).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    torch.testing.assert_close(weighted, objective_values * probability.prod(dim=-1))

    bounds = torch.stack((torch.zeros(3, dtype=torch.double), torch.ones(3, dtype=torch.double)))
    candidate, score = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="sobol",
        raw_samples=32,
        seed=1010 + q,
    )
    assert candidate.shape == (q, 3)
    assert torch.isfinite(score).all()
    assert ((candidate >= 0) & (candidate <= 1)).all()
    with torch.no_grad():
        strength, conductivity, true_probability = evaluate_truth(candidate)
    assert torch.isfinite(strength).all()
    assert torch.isfinite(conductivity).all()
    assert ((true_probability >= 0) & (true_probability <= 1)).all()
