"""Phase 8 native probability-of-improvement acquisition contracts."""

import inspect

import torch
from botorch.acquisition.monte_carlo import qProbabilityOfImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.acquisition.capabilities import AcquisitionPurpose, PosteriorRequirement
from robotorchan.acquisition.compatibility import check_model_acquisition_compatibility
from robotorchan.acquisition.optimizer_compatibility import (
    CandidateOptimizer,
    check_acquisition_optimizer_compatibility,
)
from robotorchan.acquisition.registry import ACQUISITION_REGISTRY
from robotorchan.models import SingleTaskGP
from robotorchan.optim.dispatch import optimize_acqf
from robotorchan.workflow.problem import ProblemPurpose, ProblemSpec
from robotorchan.workflow.recommendation import recommend_compatible_workflows


def test_qpi_registry_capabilities_and_native_signature() -> None:
    """Metadata covers upstream qPI without introducing a local acquisition alias."""
    entry = ACQUISITION_REGISTRY["qProbabilityOfImprovement"]
    assert entry.capabilities.purpose is AcquisitionPurpose.BAYESIAN_OPTIMIZATION
    assert entry.capabilities.posterior_requirement is PosteriorRequirement.POSTERIOR_SAMPLES
    assert entry.capabilities.max_q is None
    assert entry.capabilities.supports_constraints
    assert not entry.capabilities.one_shot
    assert entry.implementation_strategy.startswith("BoTorch-native")
    assert qProbabilityOfImprovement.__module__.startswith("botorch.")
    assert {"model", "best_f", "sampler"} <= set(
        inspect.signature(qProbabilityOfImprovement).parameters
    )
    assert check_model_acquisition_compatibility(
        "SingleTaskGP", "qProbabilityOfImprovement"
    ).compatible
    assert check_acquisition_optimizer_compatibility(
        "qProbabilityOfImprovement", CandidateOptimizer.CONTINUOUS, q=2
    ).compatible


def test_qpi_workflow_includes_supported_outcome_constraints() -> None:
    ordinary = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION)
    constrained = ProblemSpec(purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION, constrained=True)
    normal = {r.acquisition_name for r in recommend_compatible_workflows(ordinary)}
    filtered = {r.acquisition_name for r in recommend_compatible_workflows(constrained)}
    assert "qProbabilityOfImprovement" in normal
    assert "qProbabilityOfImprovement" in filtered


def test_native_qpi_evaluation_and_q_batch_candidate_optimization() -> None:
    """Execute posterior, MC sampling, acquisition and optimizer together."""
    train_X = torch.linspace(0.1, 0.9, 9, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 4)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=80)
    acquisition = qProbabilityOfImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=sampler,
    )
    X = train_X[:6].reshape(3, 2, 1)
    with torch.no_grad():
        scores = acquisition(X)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()

    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    candidates, value = optimize_acqf(
        acquisition,
        bounds,
        q=2,
        optimizer="sobol",
        raw_samples=32,
        seed=80,
    )
    assert candidates.shape == (2, 1)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()


def test_native_qpi_outcome_constraints_with_q_batch() -> None:
    """Check native qPI outcome feasibility without input-space constraints."""
    train_X = torch.linspace(0.1, 0.9, 9, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 4)
    model = SingleTaskGP(train_X, train_Y)
    model.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([32]), seed=81)

    def outcome_constraint(samples: torch.Tensor) -> torch.Tensor:
        return samples[..., 0] - 0.5

    acquisition = qProbabilityOfImprovement(
        model=model,
        best_f=train_Y.max(),
        sampler=sampler,
        constraints=[outcome_constraint],
    )
    batch = train_X[:6].reshape(3, 2, 1)
    with torch.no_grad():
        values = acquisition(batch)
    assert values.shape == (3,)
    assert torch.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
