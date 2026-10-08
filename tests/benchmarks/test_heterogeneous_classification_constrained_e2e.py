"""Phase 8: fitted regression objectives with learned binary feasibility constraints."""

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.classification_constraints import FeasibilityWeightedAcquisition
from robotorchan.acquisition.composition import (
    make_deterministic_pof_acquisition,
    make_multiple_learned_constrained_qnei_acquisition,
    make_qei_acquisition,
    resolve_acquisition_composition,
)
from robotorchan.benchmarks.heterogeneous_synthetic import evaluate_truth, observe
from robotorchan.models.classification import BinarySingleTaskGPClassifier
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics import ClassificationConstraint, ProblemSemantics, RegressionObjective


def _fitted_problem() -> tuple[HeterogeneousModel, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(808)
    train_X = torch.rand(24, 3, generator=generator, dtype=torch.double)
    observed = observe(train_X, seed=1808)
    train_Y = observed.strength.unsqueeze(-1)
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
    return (
        HeterogeneousModel(regression, classifier, output_names=["strength", "pass"]),
        train_X,
        train_Y,
    )


@pytest.mark.parametrize("kind", ["qei", "qnei"])
@pytest.mark.parametrize("q", [1, 2])
def test_classification_constrained_candidate_generation(kind: str, q: int) -> None:
    """Compose learned Pass probability with fitted qEI/qNEI and select candidates."""
    model, train_X, train_Y = _fitted_problem()
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"),),
        constraints=(ClassificationConstraint("pass", feasible_class=1),),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=808)
    if kind == "qei":
        objective_semantics = ProblemSemantics(objectives=(RegressionObjective("strength"),))
        objective_acquisition = make_qei_acquisition(
            model,
            objective_semantics,
            best_f=train_Y.max().item(),
            sampler=sampler,
        )
        binding = resolve_acquisition_composition(model, semantics).feasibility[0]
        acquisition = make_deterministic_pof_acquisition(objective_acquisition, binding)
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
    bounds = torch.stack((torch.zeros(3, dtype=torch.double), torch.ones(3, dtype=torch.double)))
    X = torch.rand(2, q, 3, generator=torch.Generator().manual_seed(q), dtype=torch.double)
    with torch.no_grad():
        objective_values = acquisition.objective_acquisition(X)
        probabilities = acquisition.probability_of_feasibility(X)
        values = acquisition(X)
    assert values.shape == (2,)
    if q == 1 and probabilities.shape == (2,):
        probabilities = probabilities.unsqueeze(-1)
    assert probabilities.shape == (2, q)
    assert torch.isfinite(values).all()
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    torch.testing.assert_close(values, objective_values * probabilities.prod(dim=-1))

    candidate, score = optimize_acqf(
        acquisition,
        bounds,
        q=q,
        optimizer="sobol",
        raw_samples=32,
        seed=808 + q,
    )
    assert candidate.shape == (q, 3)
    assert torch.isfinite(score).all()
    assert torch.all((candidate >= 0) & (candidate <= 1))
    with torch.no_grad():
        strength, _, true_probability = evaluate_truth(candidate)
    assert torch.isfinite(strength).all()
    assert torch.all((true_probability >= 0) & (true_probability <= 1))


def test_classification_feasibility_does_not_use_latent_gp_mean() -> None:
    """Verify the acquisition factor uses posterior predictive class probability."""
    model, _, _ = _fitted_problem()
    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"),),
        constraints=(ClassificationConstraint("pass", feasible_class=1),),
    )
    binding = resolve_acquisition_composition(model, semantics).feasibility[0]
    X = torch.tensor([[[0.25, 0.5, 0.75]], [[0.8, 0.2, 0.6]]], dtype=torch.double)
    with torch.no_grad():
        actual = binding.representation.probability(X)
        expected = model.entry_predict_proba(1, X)[..., 1]
    torch.testing.assert_close(actual, expected)
