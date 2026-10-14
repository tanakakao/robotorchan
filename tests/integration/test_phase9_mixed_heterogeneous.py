"""Phase 9 integration of mixed categorical regression and binary feasibility."""

import pytest
import torch
from botorch.optim import optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.classification_constraints import FeasibilityWeightedAcquisition
from robotorchan.acquisition.composition import make_mixed_constrained_qei_acquisition
from robotorchan.models.classification import MixedBinarySingleTaskGPClassifier
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import MixedSingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics import (
    ClassificationConstraint,
    ContinuousConstraint,
    ProblemSemantics,
    RegressionObjective,
)


@pytest.mark.parametrize("q", [1, 2])
def test_mixed_gp_binary_classifier_constraints_and_candidate_generation(q: int) -> None:
    """Keep discrete features valid across heterogeneous posterior semantics."""
    x = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    categories = torch.tensor([[0.0], [1.0]] * 4, dtype=torch.double)
    train_X = torch.cat((x, categories), dim=-1)
    strength = torch.sin(3 * x) + 0.1 * categories
    conductivity = torch.cos(2 * x) + 0.2 * categories
    train_Y = torch.cat((strength, conductivity), dim=-1)
    passed = (train_X[:, 0] + 0.2 * train_X[:, 1] > 0.5).to(dtype=torch.double)

    regression = MixedSingleTaskGP(train_X, train_Y, cat_dims=[1])
    classifier = MixedBinarySingleTaskGPClassifier(
        train_X, passed, cat_dims=[1], inducing_points=6
    )
    regression.eval()
    classifier.eval()
    classifier.likelihood.eval()
    model = HeterogeneousModel(
        regression, classifier, output_names=["strength", "conductivity", "pass"]
    )
    assert model.num_outputs == 3
    assert not hasattr(model, "posterior")

    semantics = ProblemSemantics(
        objectives=(RegressionObjective("strength"),),
        constraints=(
            ContinuousConstraint("conductivity", threshold=1.5),
            ClassificationConstraint("pass", feasible_class=1),
        ),
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]), seed=909)
    acquisition = make_mixed_constrained_qei_acquisition(
        model, semantics, best_f=strength.max().item(), sampler=sampler
    )
    assert isinstance(acquisition, FeasibilityWeightedAcquisition)

    candidate_X = train_X[:2].unsqueeze(1).expand(-1, q, -1).clone()
    objective_score = acquisition.objective_acquisition(candidate_X)
    feasibility = acquisition.probability_of_feasibility(candidate_X)
    score = acquisition(candidate_X)
    assert objective_score.shape == (2,)
    assert feasibility.shape == (2, q)
    assert score.shape == (2,)
    assert torch.isfinite(score).all()
    assert ((feasibility >= 0) & (feasibility <= 1)).all()
    torch.testing.assert_close(score, objective_score * feasibility.prod(dim=-1))

    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    if q == 1:
        candidates, best_score = optimize_acqf_mixed(
            acq_function=acquisition,
            bounds=bounds,
            q=q,
            fixed_features_list=[{1: 0.0}, {1: 1.0}],
            num_restarts=2,
            raw_samples=8,
            options={"maxiter": 5},
        )
    else:
        candidates, best_score = optimize_acqf(
            acquisition,
            bounds,
            q=q,
            optimizer="sobol",
            fixed_features={1: 0.0},
            raw_samples=16,
            seed=909,
        )
    assert candidates.shape == (q, 2)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(best_score).all()
    assert candidates[:, 1].tolist() in ([0.0] * q, [1.0] * q)
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()
