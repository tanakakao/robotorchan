"""Phase 6: numerical acquisition composition on the heterogeneous benchmark."""

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)

from robotorchan.acquisition.composition import (
    make_qehvi_acquisition,
    make_qei_acquisition,
    make_qnei_acquisition,
    resolve_acquisition_composition,
)
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.semantics import (
    ClassificationConstraint,
    ProblemSemantics,
    RegressionObjective,
)


def _benchmark_models() -> tuple[HeterogeneousModel, torch.Tensor]:
    generator = torch.Generator().manual_seed(606)
    train_X = torch.rand(20, 3, generator=generator, dtype=torch.double)
    observed = observe(train_X, seed=606)
    regression = SingleTaskGP(
        train_X,
        torch.stack((observed.strength, observed.conductivity), dim=-1),
    )
    classifier = BinarySingleTaskGPClassifier(train_X, observed.passed, inducing_points=8)
    model = HeterogeneousModel(
        regression, classifier, output_names=["strength", "conductivity", "pass"]
    )
    model.eval()
    return model, train_X


def test_composition_resolves_benchmark_objectives_and_feasibility() -> None:
    """Retain both regression outputs and classifier feasibility separately."""
    model, _ = _benchmark_models()
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"), RegressionObjective("conductivity")),
        constraints=(ClassificationConstraint("pass", feasible_class=1),),
    )
    plan = resolve_acquisition_composition(model, semantics)

    objective_owners = [
        (item.output_index, item.entry_index, item.local_output_index)
        for item in plan.objectives
    ]
    feasibility_owners = [
        (item.output_index, item.entry_index, item.local_output_index)
        for item in plan.feasibility
    ]
    assert objective_owners == [(0, 0, 0), (1, 0, 1)]
    assert feasibility_owners == [(2, 1, 0)]


@pytest.mark.parametrize("q", [1, 3])
def test_qei_and_qnei_produce_finite_batched_values(q: int) -> None:
    """Evaluate native BoTorch acquisitions through the semantic composition API."""
    model, train_X = _benchmark_models()
    semantics = ProblemSemantics(objectives=(RegressionObjective("strength"),))
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=606)
    best_f = evaluate_truth(train_X)[0].max().item()
    qei = make_qei_acquisition(model, semantics, best_f=best_f, sampler=sampler)
    qnei = make_qnei_acquisition(
        model,
        semantics,
        X_baseline=train_X[:6],
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=607),
        cache_root=False,
    )
    X = torch.rand(2, q, 3, generator=torch.Generator().manual_seed(q), dtype=torch.double)

    with torch.no_grad():
        ei_values = qei(X)
        nei_values = qnei(X)

    assert ei_values.shape == (2,)
    assert nei_values.shape == (2,)
    assert torch.isfinite(ei_values).all()
    assert torch.isfinite(nei_values).all()
    assert (ei_values >= 0).all()
    assert (nei_values >= 0).all()


def test_qehvi_produces_finite_batched_values() -> None:
    """Evaluate two regression objectives with a native multioutput GP posterior."""
    model, train_X = _benchmark_models()
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"), RegressionObjective("conductivity"))
    )
    strength, conductivity, _ = evaluate_truth(train_X)
    observed_Y = torch.stack((strength, conductivity), dim=-1)
    ref_point = observed_Y.min(dim=0).values - 0.2
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=observed_Y)
    acquisition = make_qehvi_acquisition(
        model,
        semantics,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=608),
    )
    X = torch.rand(2, 1, 3, generator=torch.Generator().manual_seed(608), dtype=torch.double)

    with torch.no_grad():
        values = acquisition(X)

    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    assert (values >= 0).all()
