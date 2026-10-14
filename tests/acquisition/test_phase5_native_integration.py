"""Phase 5 checks for native acquisition contracts and semantic composition."""

import inspect

import pytest
import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.composition import make_qei_acquisition
from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models.heterogeneous import HeterogeneousModel
from robotorchan.models.standard.single_task import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.semantics import ProblemSemantics, RegressionObjective


@pytest.mark.parametrize(
    ("name", "acquisition_class", "expected_parameters"),
    [
        (
            "qLogExpectedImprovement",
            qLogExpectedImprovement,
            {"model", "best_f", "sampler", "objective", "constraints"},
        ),
        (
            "qLogNoisyExpectedImprovement",
            qLogNoisyExpectedImprovement,
            {"model", "X_baseline", "sampler", "objective", "constraints"},
        ),
        (
            "qLogExpectedHypervolumeImprovement",
            qLogExpectedHypervolumeImprovement,
            {"model", "ref_point", "partitioning", "sampler"},
        ),
        (
            "qLogNoisyExpectedHypervolumeImprovement",
            qLogNoisyExpectedHypervolumeImprovement,
            {"model", "ref_point", "X_baseline", "sampler"},
        ),
        (
            "qKnowledgeGradient",
            qKnowledgeGradient,
            {"model", "num_fantasies", "sampler"},
        ),
    ],
)
def test_native_constructor_signatures_match_registry_metadata(
    name: str,
    acquisition_class: type,
    expected_parameters: set[str],
) -> None:
    """Registry entries must refer to directly usable upstream constructors."""
    entry = ACQUISITION_REGISTRY[name]
    assert entry.implementation_strategy.startswith("BoTorch-native")
    parameters = set(inspect.signature(acquisition_class).parameters)
    assert expected_parameters <= parameters
    assert entry.capabilities.max_q is None


def test_native_optimizer_accepts_q_batch_with_fixed_candidate_features() -> None:
    """Fixed candidate inputs must not be misrepresented as outcome constraints."""
    X = torch.tensor(
        [[0.1, 0.0], [0.3, 1.0], [0.5, 0.0], [0.7, 1.0], [0.9, 0.0]],
        dtype=torch.double,
    )
    Y = (X[:, :1] - 0.6).square() + X[:, 1:] * 0.2
    model = SingleTaskGP(X, Y)
    model.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=51)
    acquisition = qLogExpectedImprovement(
        model,
        best_f=Y.max().item(),
        sampler=sampler,
    )
    assert check_acquisition_optimizer_compatibility(
        "qLogExpectedImprovement", CandidateOptimizer.CONTINUOUS, q=2
    ).compatible
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    candidate, value = optimize_acqf(
        acquisition,
        bounds,
        q=2,
        optimizer="sobol",
        fixed_features={1: 1.0},
        raw_samples=32,
        seed=51,
    )
    assert candidate.shape == (2, 2)
    torch.testing.assert_close(candidate[:, 1], torch.ones(2, dtype=torch.double))
    assert torch.isfinite(value).all()


def test_heterogeneous_composition_retains_native_objective_and_q_batch() -> None:
    """The semantic bridge must produce native qEI usable by the dispatcher."""
    X = torch.linspace(0.1, 0.9, 8, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(X * 3.0)
    regression = SingleTaskGP(X, Y)
    regression.eval()
    heterogeneous = HeterogeneousModel(regression, output_names=["strength"])
    semantics = ProblemSemantics(objectives=(RegressionObjective("strength"),))
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=52)
    acquisition = make_qei_acquisition(
        heterogeneous,
        semantics,
        best_f=Y.max().item(),
        sampler=sampler,
    )
    assert acquisition.model is regression
    candidate, value = optimize_acqf(
        acquisition,
        torch.tensor([[0.0], [1.0]], dtype=torch.double),
        q=2,
        optimizer="sobol",
        raw_samples=32,
        seed=52,
    )
    assert candidate.shape == (2, 1)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()
